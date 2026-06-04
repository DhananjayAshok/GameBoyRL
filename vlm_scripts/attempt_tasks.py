"""
Called by scripts/vlm/attempt_tasks.sh (via vlm.py attempt_tasks). Use --help for CLI options.

Input
-----
A JSONL file produced by propose_tasks_zeroshot.py.
Each line: {"init_state": <str>, "tasks": [<task_str>, ...]}

Processing
----------
For each line (line_number = 0-indexed position in the JSONL):
  1. Create ONE environment for that init_state (controller_variant="low_level").
     Do not reuse or reset this env across different init_states.
  2. For each task in tasks (task_index = 0-indexed position in the task list):
       group_idx  = f"{line_number}_{task_index}"
       Reset env to init_state, then run SimpleCheckerSupervisor(
           task          = task_str,
           executor_class = <click option>,
           env           = env for this init_state,
           score_mode    = False,   # binary success/fail only
           ...
       ).evaluate()
       Collect result dict (success, description, reasoning, vlm_call_log, steps).
  3. Checkpoint after each group_idx (see below).

CLI options
-----------
  --tasks_path        Path to the input JSONL file (required)
  --executor          Short name of the executor class (default: simple)
  --max_steps         Env-step budget per task attempt (default: 50)
  --max_tool_calls    Tool-call budget per task attempt (default: 10)
  --lookback          Frames passed to checker VLM (default: 8)
  (game, model_name, vlm_kind, overwrite, verbose come from the parent click group)

Output
------
Directory: same path as tasks_path, minus extension and _attempts.
  e.g. ".../zeroshot_tasks.jsonl" -> ".../zeroshot_tasks_attempts/"

Inside that directory:
  all_trajectories.csv
    Columns: group_idx, init_state, task_string, success, n_tries
    One row per (init_state x task) attempted.

  success_trajectories.json
    dict[str, str] — group_idx -> task_string for only those that succeeded, matching the structure of infer_task.py


  success_trajectories.pkl
    dict[str, tuple] — group_idx -> (observations, actions, high_level_actions, rewards, init_state)
    observations: np.ndarray of shape (n_steps+1, H, W, C) — frame_before of step 0 + all frame_afters
    actions:            list[Type[HighLevelAction]] — one per step (executor produces high-level only)
    high_level_actions: list[tuple[Type[HighLevelAction], dict]] — (action_class, kwargs) per step
    rewards:            list[float] — one per step

Checkpointing
-------------
  checkpoint.json — dict[group_idx -> {init_state, task_string, success, description, reasoning}]
  checkpoint.pkl  — dict[group_idx -> trajectory tuple]
  On startup: if checkpoint files exist and --overwrite not set, load them and skip
  any group_idx already present.
  On completion: delete checkpoint files and write final outputs.
"""

import json
import os
import pickle

import pandas as pd

import click
import numpy as np
from tqdm import tqdm

from gameboy_worlds import get_environment
from execution.registry import AVAILABLE_EXECUTORS
from execution.report import EnvironmentStepRecord
from execution.supervisor import SimpleCheckerSupervisor
from utils import log_info, log_error
from utils.vlm import VLM


CRITIQUE_SLICE_PROMPT = """You are analysing a segment of a failed attempt to complete a task in a game of [GAME].

Task: "[TASK]"

Actions taken in this segment (steps [START_IDX]-[END_IDX] of [TOTAL] total):
[ACTION_SEQUENCE]

The images show frames [START_IDX]-[END_IDX] of the trajectory, from left to right.

Describe what happened in this segment: what the player did, what went wrong (if anything), and any observations relevant to why the task was not completed.

Respond in exactly this format:
Segment summary: <one or two sentences describing what happened in this segment>
[STOP]"""

CRITIQUE_CONSOLIDATE_PROMPT = """You are analysing a failed attempt to complete a task in a game of [GAME].

Task: "[TASK]"

Below are summaries of each segment of the failed trajectory:
[SEGMENT_SUMMARIES]

[PRIOR_HINT_BLOCK]Based on the full trajectory above, provide a concise hint for how to better approach the task on the next attempt.

Respond in exactly this format:
Critique: <what went wrong overall>
Hint: <one or two sentence hint for a better approach>
[STOP]"""


def _parse_hint(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("hint:"):
            value = stripped[len("hint:"):].strip()
            return value.replace("[STOP]", "").replace("[stop]", "").strip() or None
    return None


def _derive_hint(
    env_steps: list,
    task: str,
    game: str,
    vlm: VLM,
    max_new_tokens: int,
    previous_hint: str = "",
    max_frames_per_slice: int = 8,
) -> str:
    """Slice the failed trajectory into fixed-size windows, critique each with images,
    then consolidate into a single hint with a text-only call."""
    frames = [s.frame_after for s in env_steps]
    if not frames:
        return previous_hint

    total = len(env_steps)
    action_lines_all = [
        f"  {i + 1}. {step.action_class.get_action_name(**step.kwargs)}"
        for i, step in enumerate(env_steps)
    ]

    segment_summaries = []
    for start in range(0, total, max_frames_per_slice):
        end = min(start + max_frames_per_slice, total)
        slice_frames = frames[start:end]
        slice_actions = "\n".join(action_lines_all[start:end]) or "  (no actions taken)"
        prompt = (
            CRITIQUE_SLICE_PROMPT
            .replace("[GAME]", game)
            .replace("[TASK]", task)
            .replace("[START_IDX]", str(start + 1))
            .replace("[END_IDX]", str(end))
            .replace("[TOTAL]", str(total))
            .replace("[ACTION_SEQUENCE]", slice_actions)
        )
        output = vlm.infer(texts=prompt, images=slice_frames, max_new_tokens=max_new_tokens)
        stop_idx = output.lower().find("[stop]")
        if stop_idx != -1:
            output = output[:stop_idx]
        for line in output.splitlines():
            if line.strip().lower().startswith("segment summary:"):
                summary = line.strip()[len("segment summary:"):].strip()
                segment_summaries.append(f"Steps {start + 1}-{end}: {summary}")
                break
        else:
            segment_summaries.append(f"Steps {start + 1}-{end}: {output.strip()}")

    prior_block = (
        f'Previous hint (refine or build on this):\n"{previous_hint}"\n\n'
        if previous_hint else ""
    )
    consolidate_prompt = (
        CRITIQUE_CONSOLIDATE_PROMPT
        .replace("[GAME]", game)
        .replace("[TASK]", task)
        .replace("[SEGMENT_SUMMARIES]", "\n".join(segment_summaries))
        .replace("[PRIOR_HINT_BLOCK]", prior_block)
    )
    output = vlm.infer(texts=consolidate_prompt, max_new_tokens=max_new_tokens)
    return _parse_hint(output) or output.strip()


def _reconstruct_trajectory(env_steps: list, init_state: str) -> tuple:
    """Build a (observations, actions, high_level_actions, rewards, init_state) tuple
    from a list of EnvironmentStepRecords produced by an executor run."""
    if not env_steps:
        return None
    # observations[i] is the frame before action i; observations[-1] is the final frame
    observations = np.array(
        [env_steps[0].frame_before] + [s.frame_after for s in env_steps]
    )
    actions = [
        None for s in env_steps
    ]  # executor produces only high-level actions, so low-level is None
    high_level_actions = [(s.action_class, s.kwargs) for s in env_steps]
    rewards = [s.reward for s in env_steps]
    return (observations, actions, high_level_actions, rewards, init_state)


@click.command(name="attempt_tasks")
@click.option(
    "--tasks_path",
    required=True,
    help="Path to the JSONL produced by propose_tasks_zeroshot.",
)
@click.option(
    "--executor",
    "executor_name",
    default="simple",
    show_default=True,
    type=click.Choice(list(AVAILABLE_EXECUTORS.keys())),
    help="Executor class to use for task attempts.",
)
@click.option(
    "--max_steps",
    default=50,
    show_default=True,
    help="Env-step budget per task attempt.",
)
@click.option(
    "--max_tool_calls",
    default=10,
    show_default=True,
    help="Tool-call budget per task attempt.",
)
@click.option(
    "--lookback",
    default=8,
    show_default=True,
    help="Number of final frames passed to the checker VLM.",
)
@click.option(
    "--controller_variant",
    default="low_level",
    show_default=True,
    help="Controller variant passed to get_environment.",
)
@click.option(
    "--max_attempts",
    default=5,
    show_default=True,
    help="Maximum number of attempts per task. On each failure a hint is derived and passed to the next attempt.",
)
@click.option(
    "--checker_max_new_tokens",
    default=1000,
    show_default=True,
    help="Token budget for each checker VLM call.",
)
@click.pass_obj
def attempt_tasks_cmd(
    obj, tasks_path, executor_name, max_steps, max_tool_calls, lookback, controller_variant, max_attempts, checker_max_new_tokens
):
    """Attempt each proposed task with a VLM executor and check for success."""
    parameters = obj["parameters"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    overwrite = obj["overwrite"]
    verbose = obj["verbose"]

    executor_class = AVAILABLE_EXECUTORS[executor_name]
    max_new_tokens = obj["max_new_tokens"]
    critique_vlm = VLM(model_name, vlm_kind)

    if not os.path.exists(tasks_path):
        log_error(f"tasks_path '{tasks_path}' does not exist.", parameters)

    out_dir = os.path.splitext(tasks_path)[0] + f"_{executor_name}_attempts/"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "all_trajectories.csv")
    json_path = os.path.join(out_dir, "success_trajectories.json")
    pkl_path = os.path.join(out_dir, "success_trajectories.pkl")
    checkpoint_json = os.path.join(out_dir, "checkpoint.json")
    checkpoint_pkl = os.path.join(out_dir, "checkpoint.pkl")

    if os.path.exists(checkpoint_json) and not overwrite:
        with open(checkpoint_json, "r") as f:
            results = json.load(f)
        with open(checkpoint_pkl, "rb") as f:
            trajectories = pickle.load(f)
        log_info(
            f"Resuming from checkpoint — {len(results)} group_idxs already done.",
            parameters,
        )
    else:
        results = {}
        trajectories = {}

    with open(tasks_path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    for line_number, record in tqdm(
        enumerate(lines), total=len(lines), desc="init_states"
    ):
        init_state = record["init_state"]
        tasks = record.get("tasks", [])
        if not tasks:
            continue

        env = get_environment(
            game=game,
            controller_variant=controller_variant,
            environment_variant="default",
            init_state=init_state,
            max_steps=max_steps,
            headless=True,
            save_video=False,
        )

        for task_index, task_str in tqdm(
            enumerate(tasks), total=len(tasks), desc="tasks", leave=False
        ):
            group_idx = f"{line_number}_{task_index}"
            if group_idx in results and not overwrite:
                continue

            hint = ""
            result = None
            trajectory = None

            if verbose:
                print(f"Task [{group_idx}]: {task_str}")

            for attempt in range(max_attempts):
                env.reset()

                if verbose:
                    print(f"  Attempt {attempt + 1}/{max_attempts}" + (f" | hint: {hint}" if hint else ""))

                supervisor = SimpleCheckerSupervisor(
                    task=task_str,
                    executor_class=executor_class,
                    env=env,
                    game=game,
                    max_steps=max_steps,
                    max_tool_calls=max_tool_calls,
                    evaluation_lookback=lookback,
                    allow_self_termination=True,
                    score_mode=False,
                    checker_vlm_model=model_name,
                    checker_vlm_kind=vlm_kind,
                    checker_max_new_tokens=checker_max_new_tokens,
                    parameters=parameters,
                    vlm_model=model_name,
                    vlm_kind=vlm_kind,
                    hint=hint or None,
                )
                result = supervisor.evaluate()
                env_steps = [
                    s for s in result["steps"] if isinstance(s, EnvironmentStepRecord)
                ]
                trajectory = _reconstruct_trajectory(env_steps, init_state)

                if verbose:
                    print(f"  Result: {'success' if result['success'] else 'failure'} | {result.get('description', '')}")

                if result["success"]:
                    break

                if attempt < max_attempts - 1:
                    hint = _derive_hint(
                        env_steps, task_str, game, critique_vlm, max_new_tokens, hint
                    )
                    if verbose:
                        print(f"  Derived hint: {hint}")

            results[group_idx] = {
                "init_state": init_state,
                "task_string": task_str,
                "success": result["success"],
                "description": result.get("description", ""),
                "reasoning": result.get("reasoning", ""),
                "n_tries": attempt + 1,
            }
            trajectories[group_idx] = trajectory

            with open(checkpoint_json, "w") as f:
                json.dump(results, f, indent=2)
            with open(checkpoint_pkl, "wb") as f:
                pickle.dump(trajectories, f)

    # Write final outputs
    success_trajectories = {gid: traj for gid, traj in trajectories.items() if results.get(gid, {}).get("success")}
    with open(pkl_path, "wb") as f:
        pickle.dump(success_trajectories, f)

    pd.DataFrame([
        {
            "group_idx": group_idx,
            "init_state": res["init_state"],
            "task_string": res["task_string"],
            "success": res["success"],
            "n_tries": res["n_tries"],
        }
        for group_idx, res in results.items()
    ]).to_csv(csv_path, index=False)

    success_json = {gid: res["task_string"] for gid, res in results.items() if res.get("success")}
    with open(json_path, "w") as f:
        json.dump(success_json, f, indent=2)

    print(f"Saved results       -> {csv_path}")
    print(f"Saved success tasks -> {json_path}")
    print(f"Saved trajectories  -> {pkl_path}")

    if os.path.exists(checkpoint_json):
        os.remove(checkpoint_json)
    if os.path.exists(checkpoint_pkl):
        os.remove(checkpoint_pkl)
