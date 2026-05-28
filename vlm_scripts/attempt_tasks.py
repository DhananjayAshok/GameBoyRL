"""
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
Directory: same path as tasks_path, minus extension.
  e.g. ".../zeroshot_tasks.jsonl" -> ".../zeroshot_tasks/"

Inside that directory:
  results.csv
    Columns: group_idx, init_state, task_string, success, n_tries
    One row per (init_state x task) attempted.
    n_tries = 1 (placeholder for future retry logic).

  trajectories.pkl
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


CRITIQUE_PROMPT = """You are analysing a failed attempt to complete a task in a game of [GAME].

Task: "[TASK]"

[PRIOR_HINT_BLOCK]The images show the frames of the attempted trajectory, from left to right.

Analyse what went wrong and provide a concise hint for how to better approach the task on the next attempt.

Respond in exactly this format:
Critique: <what went wrong in this attempt>
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
) -> str:
    """Call the VLM on the failed trajectory frames to produce a hint for the next attempt."""
    frames = [s.frame_after for s in env_steps]
    if not frames:
        return previous_hint
    prior_block = (
        f'Previous hint (refine or build on this):\n"{previous_hint}"\n\n'
        if previous_hint else ""
    )
    prompt = (
        CRITIQUE_PROMPT
        .replace("[GAME]", game)
        .replace("[TASK]", task)
        .replace("[PRIOR_HINT_BLOCK]", prior_block)
    )
    output = vlm.infer(texts=prompt, images=frames, max_new_tokens=max_new_tokens)
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
    default=1,
    show_default=True,
    help="Maximum number of attempts per task. On each failure a hint is derived and passed to the next attempt.",
)
@click.pass_obj
def attempt_tasks_cmd(
    obj, tasks_path, executor_name, max_steps, max_tool_calls, lookback, controller_variant, max_attempts
):
    """Attempt each proposed task with a VLM executor and check for success."""
    parameters = obj["parameters"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    overwrite = obj["overwrite"]

    executor_class = AVAILABLE_EXECUTORS[executor_name]
    max_new_tokens = obj["max_new_tokens"]
    critique_vlm = VLM(model_name, vlm_kind)

    if not os.path.exists(tasks_path):
        log_error(f"tasks_path '{tasks_path}' does not exist.", parameters)

    out_dir = os.path.splitext(tasks_path)[0]
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results.csv")
    pkl_path = os.path.join(out_dir, "trajectories.pkl")
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

            for attempt in range(max_attempts):
                env.reset()

                supervisor = SimpleCheckerSupervisor(
                    task=task_str,
                    executor_class=executor_class,
                    env=env,
                    game=game,
                    max_steps=max_steps,
                    max_tool_calls=max_tool_calls,
                    evaluation_lookback=lookback,
                    score_mode=False,
                    checker_vlm_model=model_name,
                    checker_vlm_kind=vlm_kind,
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

                if result["success"]:
                    break

                if attempt < max_attempts - 1:
                    hint = _derive_hint(
                        env_steps, task_str, game, critique_vlm, max_new_tokens, hint
                    )

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
    with open(pkl_path, "wb") as f:
        pickle.dump(trajectories, f)

    pd.DataFrame([
        {
            "group_idx": group_idx,
            "init_state": res["init_state"],
            "task_string": res["task_string"],
            "success": res["success"],
            "n_tries": 1,
        }
        for group_idx, res in results.items()
    ]).to_csv(csv_path, index=False)

    print(f"Saved results  -> {csv_path}")
    print(f"Saved trajectories -> {pkl_path}")

    if os.path.exists(checkpoint_json):
        os.remove(checkpoint_json)
    if os.path.exists(checkpoint_pkl):
        os.remove(checkpoint_pkl)
