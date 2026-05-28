# Reads from the output JSON and pickle files produced by infer_task.py.
# For each group, generates a gold step-by-step guidance description on how
# to accomplish the inferred task, then saves the enriched data to a sibling JSON.
#
# Input files (derived from trajectory_path, same as infer_task_cmd):
#   trajectory_annotation.json  -> {group_idx: task_string}
#   trajectory_annotation.pkl   -> {group_idx: [trajectory, ...]}
#
# Output file:
#   trajectory_guidance.json    -> {group_idx: {task, init_state, guidance, goal_condition}}

import json
import os
import pickle
import click
import numpy as np
from tqdm import tqdm

from utils import log_info, log_warn, log_error
from utils.vlm import VLM
from vlm_scripts.infer_task import _parse_key, save_frames

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GUIDANCE_PROMPT = """You are an expert player of [GAME]. You are given a sequence of screenshots from a game trajectory, along with the task that was accomplished:

Task: "[TASK]"

Your job is to produce gold-standard step-by-step guidance that a new player could follow to accomplish this task from the starting frame. Base your guidance strictly on what you can observe in the frames — do not invent details not visible.

Guidelines:
- Write each step as a short, clear imperative instruction (e.g. "Press A to speak to the NPC", "Walk left towards the door").
- Order steps chronologically from start to finish.
- Be specific about directions, button presses and targets when clearly visible.
- Omit steps that are not necessary for the core task.
- If the exact button labels are intuitive, describe the action instead (e.g. "Move into the door (as opposed to interacting with it)").

Respond in exactly this format:
Summary: <one sentence describing the overall approach to complete the task. This should discuss how the frames change from left to right, leading towards task completion>
Goal condition: <a visual description of the frame at which the task is achieved. Focus specifically on what is visually present or changed in that frame that confirms the task is complete>
Steps:
- <step 1>
- <step 2>
...
[STOP]"""

# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _parse_guidance(text: str) -> dict | None:
    """Parse Summary, Goal condition, and bullet Steps from GUIDANCE_PROMPT output."""
    text_lower = text.lower()
    stop_idx = text_lower.find("[stop]")
    if stop_idx != -1:
        text_lower = text_lower[:stop_idx]

    summary = _parse_key(text_lower, "Summary") or ""
    goal_condition = _parse_key(text_lower, "Goal condition") or ""

    steps = []
    in_steps = False
    for line in text_lower.splitlines():
        stripped = line.strip()
        if stripped.startswith("steps:"):
            in_steps = True
            continue
        if in_steps and stripped.startswith("- "):
            steps.append(stripped[2:].strip())

    if not steps:
        return None
    return {"summary": summary, "goal_condition": goal_condition, "steps": steps}


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def infer_guidance_for_trajectory(
    trajectory,
    task: str,
    vlm: VLM,
    game: str,
    max_new_tokens: int,
    lookback: int = 8,
    verbose: bool = False,
) -> dict | None:
    """Given a trajectory and its task label, generate step-by-step gold guidance."""
    observations, actions, high_level_actions, rewards, init_state = trajectory
    n = len(observations)
    k = min(lookback, n)
    window = list(observations[n - k :])
    action_window = high_level_actions[n - k :]

    guidance_prompt = GUIDANCE_PROMPT.replace("[GAME]", game).replace("[TASK]", task)

    if verbose:
        save_frames(window, "guidance_window", high_level_actions=action_window)
        print(f"GUIDANCE prompt:\n{guidance_prompt}\n---")

    output = vlm.infer(
        texts=guidance_prompt,
        images=window,
        max_new_tokens=max_new_tokens,
    )

    if verbose:
        print(f"GUIDANCE output:\n{output}\n---")

    parsed = _parse_guidance(output)
    if parsed is None:
        print(f"Warning: failed to parse guidance output:\n{output}")
        return None
    return parsed


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(name="infer_guidance")
@click.option(
    "--trajectory_path",
    required=True,
    help="Path to trajectory annotation, as saved by infer_task.py",
)
@click.option(
    "--lookback",
    default=8,
    show_default=True,
    help="Number of frames from the end of each trajectory to analyse.",
)
@click.pass_obj
def infer_guidance_cmd(obj, trajectory_path, lookback):
    """Generate gold step-by-step guidance for each inferred task group."""
    max_new_tokens = obj["max_new_tokens"]
    vlm_kind = obj["vlm_kind"]
    game = obj["game"]
    model_name = obj["model_name"]
    overwrite = obj["overwrite"]
    verbose = obj["verbose"]
    parameters = obj["parameters"]

    vlm = VLM(model_name, vlm_kind)
    model_save_name = model_name.split("/")[-1]
    # have a guard to ensure we aren't using different models
    if model_save_name not in trajectory_path:
        log_error(
            f"Model name '{model_save_name}' not found in trajectory_path '{trajectory_path}'. "
            "Ensure the trajectory was produced by the same model.",
            parameters,
        )

    if not os.path.exists(trajectory_path):
        log_error(f"trajectory_path '{trajectory_path}' does not exist.", parameters)

    annotation_json = os.path.join(trajectory_path, f"trajectory_annotation.json")
    annotation_pkl = annotation_json.replace(".json", ".pkl")

    if not os.path.exists(annotation_json):
        log_error(
            f"trajectory_annotation.json not found at {annotation_json}.", parameters
        )
    if not os.path.exists(annotation_pkl):
        log_error(
            f"trajectory_annotation.pkl not found at {annotation_pkl}.", parameters
        )

    out_json = os.path.join(trajectory_path, f"trajectory_guidance.json")
    checkpoint_path = out_json.replace(".json", "_checkpoint.json")

    if os.path.exists(out_json) and not overwrite:
        log_info(
            f"Skipping infer_guidance — output already exists at {out_json}. Use --overwrite to rerun."
        )
        return

    with open(annotation_json, "r") as f:
        task_map = {int(k): v for k, v in json.load(f).items()}
    with open(annotation_pkl, "rb") as f:
        traj_map = pickle.load(f)

    if os.path.exists(checkpoint_path) and not overwrite:
        with open(checkpoint_path, "r") as f:
            guidance_output = {int(k): v for k, v in json.load(f).items()}
        log_info(
            f"Resuming infer_guidance from checkpoint — {len(guidance_output)} groups already done."
        )
    else:
        guidance_output = {}

    for group_idx in tqdm(sorted(task_map.keys()), desc="Generating guidance"):
        if group_idx in guidance_output:
            continue

        task = task_map[group_idx]
        trajectories = traj_map.get(group_idx)
        if not trajectories:
            print(f"Warning: no trajectories for group {group_idx}, skipping.")
            continue

        # Use the first available trajectory as the representative example
        trajectory = trajectories[0]
        _, _, _, _, init_state = trajectory
        guidance = infer_guidance_for_trajectory(
            trajectory,
            task,
            vlm,
            game,
            max_new_tokens,
            lookback=lookback,
            verbose=verbose,
        )
        if guidance is None:
            print(f"Warning: could not generate guidance for group {group_idx}.")
            continue

        goal_condition = guidance.pop("goal_condition", "")
        guidance_output[group_idx] = {
            "task": task,
            "init_state": init_state,
            "goal_condition": goal_condition,
            "guidance": guidance,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(guidance_output, f, indent=2)

    with open(out_json, "w") as f:
        json.dump(guidance_output, f, indent=2)
    print(f"Saved guidance annotations → {out_json}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
