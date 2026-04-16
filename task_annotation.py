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
import random
import numpy as np
import click
from PIL import Image

from utils import load_parameters
from utils.vlm import VLM, convert_numpy_greyscale_to_pillow

VERBOSE = False

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

INFER_PROMPT = """You are analysing two screenshots from a game of [GAME].

Here is a description of the sequence of frames and the changes that occured in between
[DESC_AND_CHANGES]

Over the course of some of these frames, a single primary task may have been performed by the player, with the task being completed either at the very end or in some frame close to the end. 
Describe, with a single phrase, the action or task the player performed over the course these frames? Do not use conjunctions like "and" or "while" in your description. If there are multiple distinct tasks that seem to be happening, try to describe the whole subtrajectory wholistically and omit the less important subtasks. If there is no clear task, say "NO TASK".
Be specific but concise, each task should be a single, specific and meaningful action and not trivial. Describe only what is clearly supported by the evidence above.

If there is no clear task, respond with "NO TASK"
Otherwise, respond in exactly this format:
Reasoning: <one single, short sentence describing your thinking>
Task: <one or two sentence description of what the player did or is doing>
Start: <integer index of starting frame> this is to indicate when the player seems to be moving with the intent to perform the task, not simply the step right before the task is executed. This may be several frames before the task is completed, and will depend on the specific task and context.
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

Explain in one sentence why this action was a reasonable choice given the task and what was visible on screen. Be specific and grounded in the images. Do not hallucinate.

Respond in exactly this format:
Reasoning: <one sentence explanation>
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
# Helper: high-level action class + kwargs → human-readable string
# ---------------------------------------------------------------------------

def high_level_action_to_string(high_level_action_class, kwargs: dict) -> str:
    # TODO: populate this mapping from (class, kwargs) to a descriptive string
    mapping = {}
    key = (high_level_action_class, tuple(sorted(kwargs.items())))
    if key in mapping:
        return mapping[key]
    # Fallback: ClassName(key=value, ...)
    kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return f"{high_level_action_class.__name__}({kwargs_str})"


# ---------------------------------------------------------------------------
# Parse helper: extract "Key: value" from VLM output, strip [STOP]
# ---------------------------------------------------------------------------

def _parse_key(text: str, key: str) -> str | None:
    """Return the value after 'key:' on the matching line, stripping [stop]. Case-insensitive.
    Returns None if the key is not found or the value is empty."""
    text = text.lower()
    key = key.lower()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(key + ":"):
            value = stripped[len(key) + 1:].strip()
            value = value.replace("[stop]", "").strip()
            return value if value else None
    return None


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


def _parse_infer_block(text: str) -> dict | None:
    """Parse a single Task/Start/End block from INFER output.
    Returns a dict with keys: task, start, end, or None if no Task found or task is NO TASK."""
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
        start = int(start_str)
    except (TypeError, ValueError):
        start = None
    try:
        end = int(end_str)
    except (TypeError, ValueError):
        end = None
    return {"task": task, "start": start, "end": end}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def infer_task(trajectory, vlm: VLM, game: str, max_new_tokens: int, lookback: int = 5) -> dict | None:
    """
    Three-stage pipeline: DESCRIBE (pairwise over last `lookback` frames) → INFER → REFINE.
    Returns a dict with keys: tasks (list of {task, start, end}), frame_descriptions.
    start/end are indices relative to the lookback window (0 = observations[-lookback], lookback-1 = observations[-1]).
    Returns None if parsing fails or VLM returns NO TASK.
    """
    observations, actions, high_level_actions, rewards = trajectory
    n = len(observations)
    k = min(lookback, n)
    window = observations[n - k:]  # k frames, window indices 0..k-1

    frame_descs = {}  # window index → description string
    diffs = {}        # pair index i → differences string for pair (i, i+1)

    # --- Stage 1: DESCRIBE pairwise from oldest to newest ---
    describe_prompt_template = DESCRIBE_PROMPT.replace("[GAME]", game)
    for i in range(k - 1):
        describe_output = vlm.infer(
            texts=describe_prompt_template,
            images=[window[i], window[i + 1]],
            max_new_tokens=max_new_tokens,
        ).lower()
        if VERBOSE:
            print(f"DESCRIBE output (pair {i}→{i+1}):\n{describe_output}\n---")
        frame_descs[i] = _parse_key(describe_output, "Initial Frame Description") or ""
        diffs[i] = _parse_key(describe_output, "Differences") or ""

    # Build DESC_AND_CHANGES block
    desc_lines = []
    for i in range(k - 1):
        if i == 0:
            desc_lines.append(f"Frame {i}: {frame_descs[i]}")
        desc_lines.append(f"Changes {i}→{i+1}: {diffs[i]}")
    desc_and_changes = "\n".join(desc_lines)

    # --- Stage 2: INFER (all k frames as images) ---
    infer_prompt = (
        INFER_PROMPT
        .replace("[GAME]", game)
        .replace("[DESC_AND_CHANGES]", desc_and_changes)
    )
    if VERBOSE:
        print(f"INFER prompt:\n{infer_prompt}\n---")
    infer_output = vlm.infer(
        texts=infer_prompt,
        images=list(window),
        max_new_tokens=max_new_tokens,
    ).lower()
    if VERBOSE:
        print(f"INFER output:\n{infer_output}\n---")

    if "no task" in infer_output:
        if VERBOSE:
            print("INFER returned NO TASK.")
        return None

    parsed_block = _parse_infer_block(infer_output)
    if parsed_block is None:
        print(f"Warning: infer_task failed to parse Task block from INFER stage. Output was:\n{infer_output}")
        return None

    # --- Stage 3: REFINE (text only) ---
    refine_prompt = REFINE_PROMPT.replace("[CANDIDATE_TASK]", parsed_block["task"])
    refine_output = vlm.infer(
        texts=refine_prompt,
        max_new_tokens=max_new_tokens,
    ).lower()
    if VERBOSE:
        print(f"REFINE output:\n{refine_output}\n---")
    refined_task = _parse_key(refine_output, "Task")
    if refined_task is None:
        refined_task = refine_output.lower().split("[stop]")[0].strip()  # fallback: take everything before [stop]

    return {
        "task": {"task": refined_task, "start": parsed_block["start"], "end": parsed_block["end"]},
        "frame_descriptions": frame_descs,
        "used_lookback": k,
    }


def infer_group_tasks(
    group: list,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
    lookback: int = 5,
    max_trajectories_per_group: int = 2,
) -> list[dict]:
    """
    Run infer_task on all trajectories in the group.

    Returns trajectory_data_list: per-trajectory dicts with traj_idx, tasks, frame_descriptions.
    """
    trajectory_data = []
    use_traj_idxes = list(range(len(group)))
    if len(group) > max_trajectories_per_group:
        use_traj_idxes = random.sample(use_traj_idxes, max_trajectories_per_group)
    for traj_idx in tqdm(use_traj_idxes, leave=False, total=len(use_traj_idxes), desc="Trajectories"):
        print(f"Processing trajectory {traj_idx} of {len(group)} in group...")
        trajectory = group[traj_idx]
        result = infer_task(trajectory, vlm, game, max_new_tokens, lookback)
        if result is not None:
            trajectory_data.append({
                "traj_idx": traj_idx,
                "used_lookback": result["used_lookback"],
                "task": result["task"],
                "frame_descriptions": result["frame_descriptions"],
            })
    return trajectory_data


# ---------------------------------------------------------------------------
# Click interface
# ---------------------------------------------------------------------------

@click.group()
@click.option("--model_name", required=True, help="VLM model name (e.g. gpt-4o)")
@click.option(
    "--vlm_kind",
    required=True,
    type=click.Choice(["openai", "anthropic", "openrouter", "huggingface"]),
    help="VLM backend kind",
)
@click.option("--trajectory_path", required=True, help="Path to grouped_high_reward_trajectories.pkl")
@click.option("--game", required=True, help="Game name used in prompts (e.g. 'Pokemon Red')")
@click.pass_context
def main(ctx, model_name, vlm_kind, trajectory_path, game):
    parameters = load_parameters()
    vlm = VLM(model_name, vlm_kind)
    ctx.obj = dict(
        vlm=vlm,
        trajectory_path=trajectory_path,
        game=game,
        parameters=parameters,
        model_name=model_name,
    )


@main.command()
@click.option("--max_new_tokens", default=300, show_default=True, help="Max tokens for each VLM call")
@click.option("--lookback", default=8, show_default=True, help="Number of frames from the end of each trajectory to analyse")
@click.option("--max_trajectories_per_group", default=20, show_default=True, help="Max trajectories to sample per group")
@click.pass_obj
def infer(obj, max_new_tokens, lookback, max_trajectories_per_group):
    """Infer task strings for each trajectory group."""
    vlm = obj["vlm"]
    trajectory_path = obj["trajectory_path"]
    game = obj["game"]
    model_name = obj["model_name"]
    model_save_name = model_name.split("/")[-1]

    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    trajectory_output = {}

    for group_idx, group in tqdm(enumerate(grouped_trajectories), desc="Processing groups", total=len(grouped_trajectories)):
        trajectory_data = infer_group_tasks(
            group, vlm, game, max_new_tokens, lookback, max_trajectories_per_group
        )
        if not trajectory_data:
            print(f"Warning: skipping group {group_idx} — could not infer any task strings.")
            continue
        trajectory_output[group_idx] = trajectory_data

    out_dir = os.path.dirname(trajectory_path)
    traj_path = os.path.join(out_dir, f"trajectory_annotation_{model_save_name}.json")
    with open(traj_path, "w") as f:
        json.dump(trajectory_output, f, indent=2)
    print(f"Saved trajectory annotations → {traj_path}")


@main.command()
@click.option("--safety_rollback", default=2, show_default=True, help="Extra steps before task start to include")
@click.option("--max_new_tokens", default=300, show_default=True, help="Max tokens for each VLM call")
@click.pass_obj
def reason(obj, safety_rollback, max_new_tokens):
    """Dense step-wise reasoning annotation for each task in each trajectory."""
    vlm = obj["vlm"]
    trajectory_path = obj["trajectory_path"]
    game = obj["game"]
    model_name = obj["model_name"]
    model_save_name = model_name.split("/")[-1]

    out_dir = os.path.dirname(trajectory_path)
    traj_annotation_path = os.path.join(out_dir, f"trajectory_annotation_{model_save_name}.json")
    if not os.path.exists(traj_annotation_path):
        raise FileNotFoundError(
            f"Trajectory annotation not found at {traj_annotation_path}. Run `infer` first."
        )
    with open(traj_annotation_path, "r") as f:
        trajectory_annotation = json.load(f)

    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    dense_output = {}
    for group_idx_str, traj_data_list in tqdm(trajectory_annotation.items(), desc="Processing groups"):
        group_idx = int(group_idx_str)
        group = grouped_trajectories[group_idx]

        records = []
        for traj_data in traj_data_list:
            traj_idx = traj_data["traj_idx"]
            used_lookback = traj_data["used_lookback"]
            observations, actions, high_level_actions, rewards = group[traj_idx]
            n = len(observations)
            k = used_lookback
            window_offset = n - k  # absolute index of window[0]

            task_entry = traj_data["task"]
            task_name = task_entry["task"]
            task_start = task_entry["start"]
            task_end = task_entry["end"]

            if task_start is None or task_end is None:
                continue

            step_start = max(0, task_start - safety_rollback)
            for win_step in range(step_start, task_end):
                abs_step = window_offset + win_step
                if abs_step + 1 >= n:
                    continue

                frame_t = observations[abs_step]
                frame_t1 = observations[abs_step + 1]

                action_entry = high_level_actions[abs_step] if abs_step < len(high_level_actions) else (type(None), {})
                action_class, action_kwargs = action_entry
                action_str = high_level_action_to_string(action_class, action_kwargs)

                reason_prompt = (
                    REASON_PROMPT
                    .replace("[GAME]", game)
                    .replace("[ACTION]", action_str)
                    .replace("[TASK]", task_name)
                )
                if VERBOSE:
                    print(f"REASON prompt (group {group_idx} traj {traj_idx} step {win_step}):\n{reason_prompt}\n---")
                reason_output = vlm.infer(
                    texts=reason_prompt,
                    images=[frame_t, frame_t1],
                    max_new_tokens=max_new_tokens,
                ).lower()
                if VERBOSE:
                    print(f"REASON output:\n{reason_output}\n---")
                reasoning = _parse_key(reason_output, "Reasoning")
                if reasoning is None:
                    print(f"Warning: failed to parse Reasoning for group {group_idx} traj {traj_idx} step {win_step}.")
                else:
                    refine_prompt = (
                        REASON_REFINE_PROMPT
                        .replace("[GAME]", game)
                        .replace("[CANDIDATE_REASONING]", reasoning)
                    )
                    if VERBOSE:
                        print(f"REASON_REFINE prompt:\n{refine_prompt}\n---")
                    refine_output = vlm.infer(
                        texts=refine_prompt,
                        max_new_tokens=max_new_tokens,
                    ).lower()
                    if VERBOSE:
                        print(f"REASON_REFINE output:\n{refine_output}\n---")
                    refined_reasoning = _parse_key(refine_output, "Reasoning")
                    if refined_reasoning is not None:
                        reasoning = refined_reasoning
                    else:
                        reasoning = refine_output.lower().split("[stop]")[0].strip()  # fallback

                records.append({
                    "traj_idx": traj_idx,
                    "task_name": task_name,
                    "step": win_step,
                    "action_str": action_str,
                    "reasoning": reasoning,
                })

        dense_output[group_idx] = records
        print(f"Group {group_idx}: annotated {len(records)} records.")

    out_path = os.path.join(out_dir, f"dense_annotation_{model_save_name}.json")
    with open(out_path, "w") as f:
        json.dump(dense_output, f, indent=2)
    print(f"Saved dense annotation → {out_path}")

def plot_obs_transitions(observations, save_name):
    # given list of frames (H x W) plot all the frames with horizontal concatenation as an image
    total_height = observations[0].shape[0]
    total_width = sum(obs.shape[1] for obs in observations)
    single_image = np.zeros((total_height, total_width), dtype=observations[0].dtype)
    current_x = 0
    for obs in observations:
        w = obs.shape[1]
        single_image[:, current_x:current_x+w] = obs[:, :]  # assuming (H, W, 1)
        current_x += w
    img = convert_numpy_greyscale_to_pillow(single_image)
    img.save(save_name)    



@click.command()
@click.option("--trajectory_path", required=True, help="Path to grouped_high_reward_trajectories.pkl")
@click.option("--frac", default=1.0, show_default=True, help="Fraction of observation transitions to sample")
def show(trajectory_path, frac):
    """Randomly sample a fraction of observation transitions and save them as images."""
    parameters = load_parameters()
    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    os.makedirs(parameters["tmp_dir"], exist_ok=True)
    indexes = list(range(len(grouped_trajectories)))
    random.shuffle(indexes)
    num_to_show = int(len(grouped_trajectories) * frac)
    for i, trajectory_group in tqdm(enumerate(grouped_trajectories), desc="Plotting observation transitions", total=len(grouped_trajectories)):
        if i not in indexes[:num_to_show]:
            continue
        os.makedirs(parameters["tmp_dir"] + f"/trajectories/group_{i}", exist_ok=True)
        for j, trajectory in tqdm(enumerate(trajectory_group), leave=False, total=len(trajectory_group)):
            observations = trajectory[0]            
            plot_obs_transitions(observations, parameters["tmp_dir"] + f"/trajectories/group_{i}/traj_{j}.png")
    print(f"Saved {parameters['tmp_dir']}/trajectories/")


if __name__ == "__main__":
    main()
    #show()
