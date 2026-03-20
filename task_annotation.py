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
import random

import click

from utils import load_parameters
from utils.vlm import VLM, convert_numpy_greyscale_to_pillow


# ---------------------------------------------------------------------------
# Module-level prompt constants ([GAME] is replaced at call time)
# ---------------------------------------------------------------------------

DESCRIBE_PROMPT = """You are observing two consecutive screenshots from a game of [GAME].

Image 1 is the PENULTIMATE frame. Image 2 is the FINAL frame.

Your job is to describe both frames carefully. Be precise and conservative — only state details you are confident about and that are clearly visible. Do not hallucinate or guess.

Respond in exactly this format:
Frame 1 description: <describe what is visible in the penultimate frame>
Frame 2 description: <describe what is visible in the final frame>
Similarities: <what appears in both frames>
Differences: <what has changed between frame 1 and frame 2>
[STOP]"""

INFER_PROMPT = """You are analysing two consecutive screenshots from a game of [GAME].

Here is a description of both frames:
Frame 1: [FRAME1_DESC]
Frame 2: [FRAME2_DESC]
Similarities: [SIMILARITIES]
Differences: [DIFFERENCES]

The two frames are also provided as images. Based on the descriptions and the images, answer:
What action or event did the player perform between these two frames? Or what task is being completed?

Be specific but concise. Describe only what is clearly supported by the evidence above.

Respond in exactly this format:
Task: <one or two sentence description of what the player did or is doing>
[STOP]"""

REFINE_PROMPT = """You are given a description of what a player did between two game frames:
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


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def infer_task(trajectory, vlm: VLM, game: str, max_new_tokens: int) -> str | None:
    """
    Three-stage pipeline: DESCRIBE → INFER → REFINE.
    Returns a refined imperative task string, or None if parsing fails.
    """
    observations, actions, high_level_actions, rewards = trajectory

    # TODO: check obs shape — shape is unusual: [n_timesteps, n_envs, n_steps, H, W, C]
    # The lines below are best-effort; verify with breakpoint before relying on this
    breakpoint()  # TODO: check obs shape before indexing
    final_frame = observations[-1, 0, -1]       # H x W x C
    penultimate_frame = observations[-1, 0, -2]  # H x W x C

    # --- Stage 1: DESCRIBE ---
    describe_prompt = DESCRIBE_PROMPT.replace("[GAME]", game)
    describe_output = vlm.infer(
        texts=describe_prompt,
        images=[penultimate_frame, final_frame],
        max_new_tokens=max_new_tokens,
    ).lower()

    # Best-effort: fill "" for any field that failed to parse so INFER prompt is still usable
    frame1_desc = _parse_key(describe_output, "Frame 1 description") or ""
    frame2_desc = _parse_key(describe_output, "Frame 2 description") or ""
    similarities = _parse_key(describe_output, "Similarities") or ""
    differences = _parse_key(describe_output, "Differences") or ""

    # --- Stage 2: INFER ---
    infer_prompt = (
        INFER_PROMPT
        .replace("[GAME]", game)
        .replace("[FRAME1_DESC]", frame1_desc)
        .replace("[FRAME2_DESC]", frame2_desc)
        .replace("[SIMILARITIES]", similarities)
        .replace("[DIFFERENCES]", differences)
    )
    infer_output = vlm.infer(
        texts=infer_prompt,
        images=[penultimate_frame, final_frame],
        max_new_tokens=max_new_tokens,
    ).lower()
    candidate_task = _parse_key(infer_output, "Task")

    if candidate_task is None:
        print(f"Warning: infer_task failed to parse Task from INFER stage. Output was:\n{infer_output}")
        return None

    # --- Stage 3: REFINE (text only) ---
    refine_prompt = REFINE_PROMPT.replace("[CANDIDATE_TASK]", candidate_task)
    refine_output = vlm.infer(
        texts=refine_prompt,
        max_new_tokens=max_new_tokens,
    ).lower()
    refined_task = _parse_key(refine_output, "Task")

    if refined_task is None:
        print(f"Warning: infer_task failed to parse Task from REFINE stage. Falling back to candidate.")
        return candidate_task

    return refined_task


def infer_group_task(
    group: list,
    vlm: VLM,
    game: str,
    sample_size: int,
    max_new_tokens: int,
) -> tuple[str, list[str]]:
    """
    Sample up to sample_size trajectories, infer a task string for each,
    then consolidate into a single canonical task string.

    Returns (canonical_task_string, all_candidate_strings).
    """
    sampled = random.sample(group, min(sample_size, len(group)))
    candidates = []
    for trajectory in sampled:
        task = infer_task(trajectory, vlm, game, max_new_tokens)
        if task is not None:
            candidates.append(task)

    if not candidates:
        print("Warning: all infer_task calls failed for this group. Skipping.")
        return None, []

    # Pick a random final frame for the consolidation call
    random_trajectory = random.choice(group)
    observations = random_trajectory[0]
    # TODO: check obs shape — see infer_task for shape note
    breakpoint()  # TODO: check obs shape before indexing
    random_final_frame = observations[-1, 0, -1]

    candidate_list_str = "\n".join(f"- {c}" for c in candidates)
    consolidate_prompt = (
        CONSOLIDATE_PROMPT
        .replace("[GAME]", game)
        .replace("[CANDIDATE_LIST]", candidate_list_str)
    )

    consolidate_output = vlm.infer(
        texts=consolidate_prompt,
        images=[random_final_frame],
        max_new_tokens=max_new_tokens,
    ).lower()
    canonical_task = _parse_key(consolidate_output, "Task")

    if canonical_task is None:
        print(f"Warning: failed to parse Task from CONSOLIDATE stage. Output was:\n{consolidate_output}")

    return canonical_task, candidates


def paraphrase_task_string(
    core_task_string: str,
    all_suggested_task_strings: list[str],
    vlm: VLM,
    max_new_tokens: int,
) -> list[str]:
    """
    Generate diverse imperative paraphrases of core_task_string.
    Returns a list with core_task_string as the first element.
    """
    candidate_list_str = "\n".join(f"- {s}" for s in all_suggested_task_strings)
    paraphrase_prompt = (
        PARAPHRASE_PROMPT
        .replace("[CORE_TASK]", core_task_string)
        .replace("[CANDIDATE_LIST]", candidate_list_str)
    )

    paraphrase_output = vlm.infer(
        texts=paraphrase_prompt,
        max_new_tokens=max_new_tokens,
    ).lower()
    paraphrases = _parse_bullet_list(paraphrase_output)

    return [core_task_string] + paraphrases


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
@click.option("--sample_size", default=5, show_default=True, help="Trajectories to sample per group")
@click.option("--max_new_tokens", default=300, show_default=True, help="Max tokens for each VLM call")
@click.pass_obj
def infer(obj, sample_size, max_new_tokens):
    """Infer and paraphrase task strings for each trajectory group (sparse annotation)."""
    vlm = obj["vlm"]
    trajectory_path = obj["trajectory_path"]
    game = obj["game"]
    model_name = obj["model_name"]
    model_save_name = model_name.split("/")[-1]

    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    output = {}
    for group_idx, group in grouped_trajectories.items():
        print(f"Processing group {group_idx} ({len(group)} trajectories)...")
        canonical_task, candidates = infer_group_task(
            group, vlm, game, sample_size, max_new_tokens
        )
        if canonical_task is None:
            print(f"Warning: skipping group {group_idx} — could not infer a task string.")
            continue
        task_strings = paraphrase_task_string(canonical_task, candidates, vlm, max_new_tokens)
        output[group_idx] = task_strings

    out_dir = os.path.dirname(trajectory_path)
    out_path = os.path.join(out_dir, f"task_annotation_{model_save_name}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved inferred task names → {out_path}")


@main.command()
@click.option(
    "--max_rollback",
    default=None,
    type=int,
    show_default=True,
    help="Only annotate the last N steps of each trajectory. None = full trajectory.",
)
@click.option("--max_new_tokens", default=300, show_default=True, help="Max tokens for each VLM call")
@click.pass_obj
def reason(obj, max_rollback, max_new_tokens):
    """Dense step-wise reasoning annotation for each action in each trajectory."""
    vlm = obj["vlm"]
    trajectory_path = obj["trajectory_path"]
    game = obj["game"]
    model_name = obj["model_name"]
    model_save_name = model_name.split("/")[-1]

    # Load sparse task annotation (must already exist)
    out_dir = os.path.dirname(trajectory_path)
    sparse_path = os.path.join(out_dir, f"task_annotation_{model_save_name}.json")
    if not os.path.exists(sparse_path):
        raise FileNotFoundError(
            f"Sparse annotation not found at {sparse_path}. Run `infer` first."
        )
    with open(sparse_path, "r") as f:
        task_annotation = json.load(f)

    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    dense_output = {}
    for group_idx, group in grouped_trajectories.items():
        group_key = str(group_idx)
        # canonical task string is the first element
        task_strings = task_annotation.get(group_key, task_annotation.get(group_idx, []))
        canonical_task = task_strings[0] if task_strings else ""

        records = []
        for traj_idx, trajectory in enumerate(group):
            observations, actions, high_level_actions, rewards = trajectory

            # TODO: check obs shape — shape is unusual: [n_timesteps, n_envs, n_steps, H, W, C]
            breakpoint()  # TODO: check obs shape before indexing

            n_steps = observations.shape[0]  # best-effort: treat first dim as time
            start = 0 if max_rollback is None else max(0, n_steps - max_rollback)

            for step in range(start, n_steps - 1):
                frame_t = observations[step]
                frame_t1 = observations[step + 1]

                action_entry = high_level_actions[step] if step < len(high_level_actions) else (type(None), {})
                # high_level_actions entries are expected to be (HighLevelAction class, kwargs dict)
                action_class, action_kwargs = action_entry
                action_str = high_level_action_to_string(action_class, action_kwargs)

                reason_prompt = (
                    REASON_PROMPT
                    .replace("[GAME]", game)
                    .replace("[ACTION]", action_str)
                    .replace("[TASK]", canonical_task)
                )
                reason_output = vlm.infer(
                    texts=reason_prompt,
                    images=[frame_t, frame_t1],
                    max_new_tokens=max_new_tokens,
                ).lower()
                reasoning = _parse_key(reason_output, "Reasoning")
                if reasoning is None:
                    print(f"Warning: failed to parse Reasoning for group {group_idx} traj {traj_idx} step {step}.")

                records.append(
                    {
                        "traj_idx": traj_idx,
                        "step": step,
                        "action_str": action_str,
                        "reasoning": reasoning,
                    }
                )

        dense_output[group_idx] = records
        print(f"Group {group_idx}: annotated {len(records)} steps across {len(group)} trajectories.")

    out_path = os.path.join(out_dir, f"dense_annotation_{model_save_name}.json")
    with open(out_path, "w") as f:
        json.dump(dense_output, f, indent=2)
    print(f"Saved dense annotation → {out_path}")


if __name__ == "__main__":
    main()
