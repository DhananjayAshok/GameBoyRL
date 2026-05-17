# trajectories are saved in parameters["storage_dir"]/grouped_trajectories/$game/<path>/grouped_high_reward_trajectories.pkl
# it is a dictionary with numeric keys and values that are lists of trajectories
# each trajectory is a list (observations, actions, high_level_actions, rewards)
# observations is a stack of frames (numpy array) with shape (num_frames, 144, 160, 1)
# actions is a list of integers with shape (num_frames-1,)
# high_level_actions is a list of integers with shape (num_frames-1,)
# rewards is a list of floats with shape (num_frames-1,)
# the grouping is done by similarity of the final frames of the trajectories

import json
import os
import pickle
from tqdm import tqdm
import numpy as np
import click
from PIL import Image

from utils import load_parameters, log_info
from utils.vlm import VLM
from show_trajectories import plot_transitions

SAVE_FRAMES_DIR = "save_frames"


def save_frames(frames, save_name, high_level_actions=None):
    os.makedirs(SAVE_FRAMES_DIR, exist_ok=True)
    plot_transitions(
        frames,
        os.path.join(SAVE_FRAMES_DIR, f"{save_name}.png"),
        high_level_actions=high_level_actions,
    )


# ---------------------------------------------------------------------------
# Module-level prompt constants ([GAME] is replaced at call time)
# ---------------------------------------------------------------------------

DESCRIBE_PROMPT = """You are observing two consecutive screenshots from a game of [GAME].

Image 1 is the INITIAL frame. Image 2 is the NEXT frame.

Your job is to describe both frames carefully. Be precise and conservative — only state details you are confident about and that are clearly visible. Do not hallucinate or guess.

Respond in exactly this format:
Initial Frame Description: <describe what is visible in the initial frame>
Differences: <what has changed between frame 1 and frame 2>
[STOP]"""

INFER_PROMPT = """You are analysing multiple screenshots in sequence from a game of [GAME].

[DESC_AND_CHANGES]

Over the course of some of these frames, a single primary task may have been performed by the player, with the task being completed either at the very end or in some frame close to the end. 
Describe, with a single phrase, the action or task the player performed over the course these frames? Do not use conjunctions like "and" or "while" in your description. If there are multiple distinct tasks that seem to be happening, try to describe the whole subtrajectory wholistically and omit the less important subtasks. If there is no clear task, say "NO TASK".
Be specific but concise, each task should be a single, specific and meaningful action and not trivial. Describe only what is clearly supported by the evidence above.

Always try to pick the longest horizon, most multistep version of the task that is present in the trajectory. If there is no clear task, respond with "NO TASK"
Otherwise, respond in exactly this format:
Visual Description: <a description of the individual frames and changes that occur from leftmost frame to rightmost frame>
Reasoning: <one single, short sentence describing your thinking. Reference visual evidence of the key frames and overall actions that led you to infer this task.>
Task: <one or two sentence description of what the player did or is doing>
Start: <integer index of starting frame> this is to indicate when the player seems to be acting with the intent to perform the task, not simply the step right before the task is executed. This may be several frames before the task is completed, and will depend on the specific task and context.
End: <integer index of frame where task is performed or executed or completed or first detected>
[STOP]"""

REFINE_PROMPT = """You are given a description of what a player did over the course of some game frames:
"[CANDIDATE_TASK]"

Rewrite this as a concise imperative instruction (e.g. "Walk into the building", "Open the menu", "Talk to the NPC").
- Use second-person imperative tone (no subject).
- Keep it short (under 10 words if possible).
- Do not add any detail that was not in the original description.

Respond in exactly this format:
Task: <imperative task string>
[STOP]"""

CONSOLIDATE_PROMPT = """You are given several candidate descriptions of a task performed in a game of [GAME], all inferred from similar game states:
[CANDIDATE_LIST]

A representative final frame is provided as an image.

Identify the single most accurate, concise imperative task string that best captures what all of these trajectories have in common. Ignore noise or outliers.

Respond in exactly this format:
Task: <single canonical imperative task string>
[STOP]"""

PARAPHRASE_PROMPT = """You are given a canonical task string for a game task:
"[CORE_TASK]"

Here are other phrasings that have been suggested for the same task:
[CANDIDATE_LIST]

Generate as many diverse, valid paraphrases as possible. Vary the wording, phrasing style, and level of specificity, but preserve the core meaning. Use imperative tone throughout.

Respond in exactly this format (one per line):
- <paraphrase 1>
- <paraphrase 2>
...
[STOP]"""

REASON_PROMPT = """You are observing two consecutive screenshots from a game of [GAME].

Image 1 is the frame BEFORE the action. Image 2 is the frame AFTER the action.
The action taken was: [ACTION]
The overall task being pursued is: [TASK]

Did this action enable or lead to progress towards the task in this context? Describe the screen in a task-oriented manner and then either justify why the player took this action or why the action was not relevant to the task. Be specific and grounded in the images. Do not hallucinate or guess details that are not clearly visible.

Respond in exactly this format:
Verdict: <YES or NO, indicating whether the action was relevant for task progress>
Reasoning: <very brief explanation either explaining how the action helped progress towards the task, or why it was not relevant to the task, based on the images. Make sure to explicitly refer to visual evidence>
[STOP]"""

REASON_REFINE_PROMPT = """You are given a candidate reasoning for why a particular action was taken in a game of [GAME]: 
Task: [TASK]
Action taken: [ACTION]
Candidate: [CANDIDATE_REASONING]

Refine this by writing it in the first person perspective of the player in the present tense, and making it more imperitive in nature. So use words such as "the screen shows X ..., it makes most sense to do Y, hence I should press Z ... or something like that.

Answer in exactly this format:
Reasoning: <refined reasoning that justifies exactly why the action was taken in context of the task in first person future planning language>
[STOP]
"""

# ---------------------------------------------------------------------------
# Parse helper: extract "Key: value" from VLM output, strip [STOP]
# ---------------------------------------------------------------------------


def _parse_key(text: str, key: str) -> str | None:
    """Return the value after 'key:' on the matching line, stripping [stop]. Case-insensitive.
    Returns None if the key is not found or the value is empty."""
    text = text.lower()
    # if there is only one response: , then only get the stuff after
    if text.count("response:") == 1:
        text = text.split("response:")[1].strip()
    key = key.lower()
    n_keys = text.count(f"{key}:")
    if n_keys == 0:
        n_key_mentions = text.count(key)
        if n_key_mentions == 1:
            return text.split(key)[1].splitlines()[0].strip().replace("[stop]", "")
        else:
            return None
    elif n_keys == 1:
        return text.split(f"{key}:")[1].splitlines()[0].strip().replace("[stop]", "")
    else:  # see if maybe only one of them starts with key, and then return that
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(key + ":"):
                value = stripped[len(key) + 1 :].strip()
                value = value.replace("[stop]", "").strip()
                return value if value else None


def _parse_bullet_list(text: str) -> list[str]:
    """Return all '- ...' bullet lines before [stop]. Case-insensitive."""
    results = []
    for line in text.lower().splitlines():
        if "[stop]" in line:
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            results.append(stripped[2:].strip())
    return results


def _parse_infer_block(
    text: str, window_offset: int = 0, n_obs: int = None
) -> dict | None:
    """Parse a single Task/Start/End block from INFER output.
    Returns a dict with keys: task, start, end as absolute obs array indices, or None if no Task found or task is NO TASK.
    """
    text = text.lower()
    stop_idx = text.find("[stop]")
    if stop_idx != -1:
        text = text[:stop_idx]

    task = _parse_key(text, "Task")
    if task is None or "no task" in task:
        return None
    start_str = _parse_key(text, "Start")
    end_str = _parse_key(text, "End")
    try:
        start = int(start_str) - 1 + window_offset
    except (TypeError, ValueError):
        start = None
    try:
        end = int(end_str) - 1 + window_offset
        if n_obs is not None and end >= n_obs - 1:
            end -= 1
    except (TypeError, ValueError):
        end = None
    if n_obs is not None:
        if start is not None:
            start = max(0, min(start, n_obs - 1))
        if end is not None:
            end = max(0, min(end, n_obs - 2))
    return {"task": task, "start": start, "end": end}


def got_bigger(subject, refined_subject, multiplier=1.5):
    if refined_subject is None:
        return None
    subject_count = subject.count(" ")
    refined_count = refined_subject.count(" ")
    if refined_count > multiplier * subject_count:
        return True
    else:
        return False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def describe_pairwise(
    window,
    action_window,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
    verbose: bool = False,
) -> str:
    """Run pairwise DESCRIBE over consecutive frame pairs and build a DESC_AND_CHANGES block."""
    k = len(window)
    frame_descs = {}
    diffs = {}
    describe_prompt_template = DESCRIBE_PROMPT.replace("[GAME]", game)
    for i in range(k - 1):
        describe_output = vlm.infer(
            texts=describe_prompt_template,
            images=[window[i], window[i + 1]],
            max_new_tokens=max_new_tokens,
        ).lower()
        if verbose:
            save_frames(
                [window[i], window[i + 1]],
                f"describe_pair_{i}",
                high_level_actions=[action_window[i]],
            )
            print(f"DESCRIBE output (pair {i}→{i+1}):\n{describe_output}\n---")
        frame_descs[i] = _parse_key(describe_output, "Initial Frame Description") or ""
        diffs[i] = _parse_key(describe_output, "Differences") or ""

    desc_lines = []
    for i in range(k - 1):
        if i == 0:
            desc_lines.append(f"Frame {i}: {frame_descs[i]}")
        desc_lines.append(f"Changes {i}→{i+1}: {diffs[i]}")
    if desc_lines:
        return (
            "Here is a description of the changes that occured in between each frame:\n"
            + "\n".join(desc_lines),
            frame_descs,
        )
    return "", frame_descs


def infer_task(
    trajectory,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
    lookback: int = 5,
    verbose: bool = False,
    describe_pairs: bool = False,
) -> dict | None:
    """
    Three-stage pipeline: DESCRIBE (pairwise over last `lookback` frames) → INFER → REFINE.
    Returns a dict with keys: tasks (list of {task, start, end}), frame_descriptions, task_description.
    start/end are absolute obs array indices.
    Returns None if parsing fails or VLM returns NO TASK.
    """
    observations, actions, high_level_actions, rewards = trajectory
    n = len(observations)
    k = min(lookback, n)
    window = observations[n - k :]  # k frames, window indices 0..k-1
    action_window = high_level_actions[
        n - k :
    ]  # k-1 actions, action_window[i] is the action causing window[i]→window[i+1]

    frame_descs = {}

    if describe_pairs:
        desc_and_changes, frame_descs = describe_pairwise(
            window, action_window, vlm, game, max_new_tokens, verbose=verbose
        )
    else:
        desc_and_changes = ""

    # --- Stage 2: INFER (all k frames as images) ---
    infer_prompt = INFER_PROMPT.replace("[GAME]", game).replace(
        "[DESC_AND_CHANGES]", desc_and_changes
    )
    if verbose:
        save_frames(list(window), "infer_window", high_level_actions=action_window)
        print(f"INFER prompt:\n{infer_prompt}\n---")
    infer_output = vlm.infer(
        texts=infer_prompt,
        images=list(window),
        max_new_tokens=max_new_tokens,
    ).lower()
    if verbose:
        print(f"INFER output:\n{infer_output}\n---")

    if "no task" in infer_output:
        if verbose:
            print("INFER returned NO TASK.")
        return None

    parsed_block = _parse_infer_block(infer_output, window_offset=n - k, n_obs=n)
    if parsed_block is None:
        print(
            f"Warning: infer_task failed to parse Task block from INFER stage. Output was:\n{infer_output}"
        )
        return None
    task_description = _parse_key(infer_output, "Visual Description")

    # --- Stage 3: REFINE (text only) ---
    refine_prompt = REFINE_PROMPT.replace("[CANDIDATE_TASK]", parsed_block["task"])
    refine_output = vlm.infer(
        texts=refine_prompt,
        max_new_tokens=max_new_tokens,
    ).lower()
    if verbose:
        print(f"REFINE output:\n{refine_output}\n---")
    refined_task = _parse_key(refine_output, "Task")
    if refined_task is None:
        refined_task = (
            refine_output.lower().split("[stop]")[0].strip()
        )  # fallback: take everything before [stop]
    task = parsed_block["task"]
    if got_bigger(task, refined_task):
        refined_task = task

    if verbose:
        print(
            f"Final inferred task: {refined_task}, start: {parsed_block['start']}, end: {parsed_block['end']}"
        )
        # save_frames of both start and end
        start_idx = parsed_block["start"] if parsed_block["start"] is not None else None
        end_idx = parsed_block["end"] if parsed_block["end"] is not None else None
        if start_idx is not None and end_idx is not None:
            save_frames([observations[start_idx], observations[end_idx]], "task_frames")
        print(
            "Exiting after one inference for verbose demonstration. Set VERBOSE = False to run full inference."
        )
        breakpoint()
    return {
        "task": {
            "task": refined_task,
            "start": parsed_block["start"],
            "end": parsed_block["end"],
        },
        "task_description": task_description,
        "frame_descriptions": frame_descs if frame_descs else None,
        "used_lookback": k,
    }


def infer_group_tasks(
    group: list,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
    lookback: int = 5,
    max_trajectories_per_group: int = 2,
    verbose: bool = False,
    describe_pairs: bool = False,
) -> list[dict]:
    """
    Run infer_task on all trajectories in the group.

    Returns trajectory_data_list: per-trajectory dicts with traj_idx, tasks, frame_descriptions.
    """
    trajectory_data = []
    use_traj_idxes = list(range(len(group)))
    if len(group) > max_trajectories_per_group:
        use_traj_idxes = list(
            np.random.choice(use_traj_idxes, max_trajectories_per_group, replace=False)
        )
    for traj_idx in tqdm(
        use_traj_idxes, leave=False, total=len(use_traj_idxes), desc="Trajectories"
    ):
        trajectory = group[traj_idx]
        result = infer_task(
            trajectory,
            vlm,
            game,
            max_new_tokens,
            lookback,
            verbose=verbose,
            describe_pairs=describe_pairs,
        )
        if result is not None:
            trajectory_data.append(
                {
                    "traj_idx": int(
                        traj_idx
                    ),  # np.random.choice returns np.int64, not JSON serializable
                    "used_lookback": result["used_lookback"],
                    "task": result["task"],
                    "task_description": result["task_description"],
                    "frame_descriptions": result["frame_descriptions"],
                }
            )
    return trajectory_data


# ---------------------------------------------------------------------------
# Click interface
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--lookback",
    default=8,
    show_default=True,
    help="Number of frames from the end of each trajectory to analyse",
)
@click.option(
    "--max_trajectories_per_group",
    default=20,
    show_default=True,
    help="Max trajectories to sample per group",
)
@click.option(
    "--describe_pairs",
    is_flag=True,
    default=False,
    help="Run pairwise DESCRIBE stage before INFER.",
)
@click.pass_obj
def infer(obj, lookback, max_trajectories_per_group, describe_pairs):
    """Infer task strings for each trajectory group."""
    max_new_tokens = obj["max_new_tokens"]
    vlm_kind = obj["vlm_kind"]
    trajectory_path = obj["trajectory_path"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm = VLM(model_name, vlm_kind)
    model_save_name = model_name.split("/")[-1]

    out_dir = os.path.dirname(trajectory_path)
    traj_path = os.path.join(out_dir, f"trajectory_annotation_{model_save_name}.json")
    checkpoint_path = traj_path.replace(".json", "_checkpoint.json")

    if os.path.exists(traj_path) and not obj["overwrite"]:
        log_info(
            f"Skipping infer — output already exists at {traj_path}. Use --overwrite to rerun."
        )
        return

    if os.path.exists(checkpoint_path) and not obj["overwrite"]:
        with open(checkpoint_path, "r") as f:
            trajectory_output = {int(k): v for k, v in json.load(f).items()}
        log_info(
            f"Resuming infer from checkpoint — {len(trajectory_output)} groups already done."
        )
    else:
        trajectory_output = {}

    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    for group_idx, group in tqdm(
        enumerate(grouped_trajectories),
        desc="Processing groups",
        total=len(grouped_trajectories),
    ):
        if group_idx in trajectory_output:
            continue
        trajectory_data = infer_group_tasks(
            group,
            vlm,
            game,
            max_new_tokens,
            lookback,
            max_trajectories_per_group,
            verbose=obj["verbose"],
            describe_pairs=describe_pairs,
        )
        if not trajectory_data:
            print(
                f"Warning: skipping group {group_idx} — could not infer any task strings."
            )
            continue
        trajectory_output[group_idx] = trajectory_data
        with open(checkpoint_path, "w") as f:
            json.dump(trajectory_output, f, indent=2)

    with open(traj_path, "w") as f:
        json.dump(trajectory_output, f, indent=2)
    print(f"Saved trajectory annotations → {traj_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


@click.command()
@click.option(
    "--safety_rollback",
    default=2,
    show_default=True,
    help="Extra steps before task start to include",
)
@click.pass_obj
def reason(obj, safety_rollback):
    """Dense step-wise reasoning annotation for each task in each trajectory."""
    max_new_tokens = obj["max_new_tokens"]
    vlm_kind = obj["vlm_kind"]
    trajectory_path = obj["trajectory_path"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm = VLM(model_name, vlm_kind)
    model_save_name = model_name.split("/")[-1]
    verbose = obj["verbose"]

    out_dir = os.path.dirname(trajectory_path)
    out_path = os.path.join(out_dir, f"dense_annotation_{model_save_name}.json")
    checkpoint_path = out_path.replace(".json", "_checkpoint.json")

    if os.path.exists(out_path) and not obj["overwrite"]:
        log_info(
            f"Skipping reason — output already exists at {out_path}. Use --overwrite to rerun."
        )
        return

    if os.path.exists(checkpoint_path) and not obj["overwrite"]:
        with open(checkpoint_path, "r") as f:
            dense_output = {int(k): v for k, v in json.load(f).items()}
        log_info(
            f"Resuming reason from checkpoint — {len(dense_output)} groups already done."
        )
    else:
        dense_output = {}

    traj_annotation_path = os.path.join(
        out_dir, f"trajectory_annotation_{model_save_name}.json"
    )
    if not os.path.exists(traj_annotation_path):
        raise FileNotFoundError(
            f"Trajectory annotation not found at {traj_annotation_path}. Run `infer` first."
        )
    with open(traj_annotation_path, "r") as f:
        trajectory_annotation = json.load(f)

    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    for group_idx_str, traj_data_list in tqdm(
        trajectory_annotation.items(), desc="Processing groups"
    ):
        group_idx = int(group_idx_str)
        if group_idx in dense_output:
            continue
        group = grouped_trajectories[group_idx]

        records = []
        for traj_data in traj_data_list:
            traj_idx = traj_data["traj_idx"]
            observations, actions, high_level_actions, rewards = group[traj_idx]
            n = len(observations)

            task_entry = traj_data["task"]
            task_name = task_entry["task"]
            task_start = task_entry["start"]
            task_end = task_entry["end"]

            if task_start is None or task_end is None:
                continue

            step_start = max(0, task_start - safety_rollback)
            for abs_step in range(step_start, task_end + 1):
                if abs_step + 1 >= n:
                    continue

                frame_t = observations[abs_step]
                frame_t1 = observations[abs_step + 1]

                action_entry = (
                    high_level_actions[abs_step]
                    if abs_step < len(high_level_actions)
                    else (type(None), {})
                )
                action_class, action_kwargs = action_entry
                action_str = (
                    action_class.get_action_name(**action_kwargs)
                    if action_class is not None
                    else "NO ACTION"
                )

                reason_prompt = (
                    REASON_PROMPT.replace("[GAME]", game)
                    .replace("[ACTION]", action_str)
                    .replace("[TASK]", task_name)
                )
                if verbose:
                    save_frames(
                        [frame_t, frame_t1],
                        f"reason_g{group_idx}_t{traj_idx}_s{abs_step}",
                        high_level_actions=[action_entry],
                    )
                    print(
                        f"REASON prompt (group {group_idx} traj {traj_idx} step {abs_step}):\n{reason_prompt}\n---"
                    )
                reason_output = vlm.infer(
                    texts=reason_prompt,
                    images=[frame_t, frame_t1],
                    max_new_tokens=max_new_tokens,
                ).lower()
                if verbose:
                    print(f"REASON output:\n{reason_output}\n---")
                verdict_str = _parse_key(reason_output, "Verdict")
                good_action = (
                    verdict_str is not None and verdict_str.strip().startswith("yes")
                )
                reasoning = _parse_key(reason_output, "Reasoning")

                if reasoning is None:
                    print(
                        f"Warning: failed to parse Reasoning for group {group_idx} traj {traj_idx} step {abs_step}."
                    )
                elif good_action:
                    refine_prompt = (
                        REASON_REFINE_PROMPT.replace("[GAME]", game)
                        .replace("[TASK]", task_name)
                        .replace("[ACTION]", action_str)
                        .replace("[CANDIDATE_REASONING]", reasoning)
                    )
                    if verbose:
                        print(f"REASON_REFINE prompt:\n{refine_prompt}\n---")
                    refine_output = vlm.infer(
                        texts=refine_prompt,
                        max_new_tokens=max_new_tokens,
                    ).lower()
                    if verbose:
                        print(f"REASON_REFINE output:\n{refine_output}\n---")
                    refined_reasoning = _parse_key(refine_output, "Reasoning")
                    if refined_reasoning is None:
                        refined_reasoning = (
                            refine_output.lower().split("[stop]")[0].strip()
                        )
                    if got_bigger(reasoning, refined_reasoning, multiplier=3.0):
                        pass
                    else:
                        reasoning = refined_reasoning

                if verbose:
                    print(
                        f"Final reasoning (good_action={good_action}) for group {group_idx} traj {traj_idx} step {abs_step}:\n{reasoning}\n==="
                    )
                records.append(
                    {
                        "traj_idx": traj_idx,
                        "task_name": task_name,
                        "step": abs_step,
                        "action_str": action_str,
                        "reasoning": reasoning,
                        "good_action": good_action,
                    }
                )
            if verbose:
                print(
                    "Exiting after one reasoning for verbose demonstration. Set VERBOSE = False to run full annotation."
                )
                breakpoint()

        dense_output[group_idx] = records
        with open(checkpoint_path, "w") as f:
            json.dump(dense_output, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(dense_output, f, indent=2)
    print(f"Saved dense annotation → {out_path}")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
