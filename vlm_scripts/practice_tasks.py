"""
Input
-----
A trajectory_guidance.json produced by infer_guidance.py.
Format: {group_idx: {task, init_state, goal_condition, guidance}}
  guidance: {"summary": <str>, "steps": [<str>, ...]}

Processing
----------
For each group_idx:
  1. Create one environment for the group's init_state.
  2. For each attempt in range(n_attempts):
       a. Seed numpy with (base_seed + hash(group_idx) + attempt) for reproducibility.
       b. Reset env, then take n_random_actions random low-level steps to perturb start state.
       c. Run SimpleCheckerSupervisor with task, guidance (formatted string), and goal_condition.
          Score mode is a click option.
       d. Save the vlm_call_log from the result to a per-episode pickle.
  3. No hint derivation between attempts — each is independent.

CLI options
-----------
  --guidance_path     Path to trajectory_guidance.json (required)
  --n_attempts        Number of independent attempts per task (default: 3)
  --n_random_actions  Random env steps before each attempt (default: 5)
  --score_mode        Flag: use 1-10 score instead of binary success
  --max_steps         Env-step budget per attempt (default: 50)
  --max_tool_calls    Tool-call budget per attempt (default: 10)
  --lookback          Frames passed to checker VLM (default: 8)
  --executor          Executor class short name (default: simple)
  --controller_variant  (default: low_level)

Output
------
Directory: same directory as guidance_path, under a "practice/" subfolder.

  practice/{group_idx}_{attempt}.pkl
    List[VLMCallRecord] — all VLM calls made during that episode.

  practice/results.csv
    Columns: group_idx, attempt, task_string, success, score
    success is NaN when score_mode=True.
    score   is NaN when score_mode=False.

Checkpointing
-------------
  practice/checkpoint.json — list of completed result rows.
  On startup: load completed rows, skip already-done (group_idx, attempt) pairs.
  On completion: write final CSV, delete checkpoint.
"""

import json
import os
import pickle

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from gameboy_worlds import get_environment
from execution.registry import AVAILABLE_EXECUTORS
from execution.supervisor import SimpleCheckerSupervisor
from utils import log_info, log_error


def _format_guidance(guidance_dict: dict) -> str:
    """Format a guidance dict {summary, steps} into a readable string for the supervisor."""
    lines = []
    if guidance_dict.get("summary"):
        lines.append(f"Summary: {guidance_dict['summary']}")
    steps = guidance_dict.get("steps", [])
    if steps:
        lines.append("Steps:")
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
    return "\n".join(lines)


def _episode_seed(base_seed: int, group_idx: str, attempt: int) -> int:
    return base_seed + hash(group_idx) % (2**16) + attempt


@click.command(name="practice_tasks")
@click.option(
    "--guidance_path",
    required=True,
    help="Path to trajectory_guidance.json produced by infer_guidance.",
)
@click.option(
    "--n_attempts",
    default=3,
    show_default=True,
    help="Number of independent attempts per task.",
)
@click.option(
    "--n_random_actions",
    default=5,
    show_default=True,
    help="Random low-level env steps taken before each attempt to perturb start state.",
)
@click.option(
    "--score_mode",
    is_flag=True,
    default=False,
    help="Evaluate with a 1-10 score instead of binary success.",
)
@click.option(
    "--max_steps",
    default=50,
    show_default=True,
    help="Env-step budget per attempt.",
)
@click.option(
    "--max_tool_calls",
    default=10,
    show_default=True,
    help="Tool-call budget per attempt.",
)
@click.option(
    "--lookback",
    default=8,
    show_default=True,
    help="Number of final frames passed to the checker VLM.",
)
@click.option(
    "--executor",
    "executor_name",
    default="simple",
    show_default=True,
    type=click.Choice(list(AVAILABLE_EXECUTORS.keys())),
    help="Executor class to use.",
)
@click.option(
    "--controller_variant",
    default="low_level",
    show_default=True,
    help="Controller variant passed to get_environment.",
)
@click.option(
    "--checker_max_new_tokens",
    default=1000,
    show_default=True,
    help="Token budget for each checker VLM call.",
)
@click.pass_obj
def practice_tasks_cmd(
    obj,
    guidance_path,
    n_attempts,
    n_random_actions,
    score_mode,
    max_steps,
    max_tool_calls,
    lookback,
    executor_name,
    controller_variant,
    checker_max_new_tokens,
):
    """Run repeated supervised practice attempts on inferred tasks."""
    parameters = obj["parameters"]
    game = obj["game"]
    model_name = obj["model_name"]
    vlm_kind = obj["vlm_kind"]
    overwrite = obj["overwrite"]
    verbose = obj["verbose"]
    base_seed = parameters["random_seed"]

    executor_class = AVAILABLE_EXECUTORS[executor_name]

    if not os.path.exists(guidance_path):
        log_error(f"guidance_path '{guidance_path}' does not exist.", parameters)

    out_dir = os.path.join(os.path.dirname(guidance_path), "practice")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results.csv")
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")

    if os.path.exists(checkpoint_path) and not overwrite:
        with open(checkpoint_path, "r") as f:
            rows = json.load(f)
        done = {(r["group_idx"], r["attempt"]) for r in rows}
        log_info(f"Resuming from checkpoint — {len(rows)} episodes already done.", parameters)
    else:
        rows = []
        done = set()

    with open(guidance_path, "r") as f:
        guidance_data = {str(k): v for k, v in json.load(f).items()}

    for group_idx, record in tqdm(guidance_data.items(), desc="groups"):
        task_str = record["task"]
        init_state = record["init_state"]
        goal_condition = record.get("goal_condition", "") or None
        guidance_str = _format_guidance(record.get("guidance", {}))

        if verbose:
            print(f"Group [{group_idx}] task: {task_str}")
            if guidance_str:
                print(f"  Guidance: {guidance_str}")

        env = get_environment(
            game=game,
            controller_variant=controller_variant,
            init_state=init_state,
            max_steps=max_steps,
            headless=True,
            save_video=False,
        )

        for attempt in tqdm(range(n_attempts), desc="attempts", leave=False):
            if (group_idx, attempt) in done:
                continue

            np.random.seed(_episode_seed(base_seed, group_idx, attempt))
            env.reset()
            for _ in range(n_random_actions):
                env.step(env.action_space.sample())

            supervisor = SimpleCheckerSupervisor(
                task=task_str,
                executor_class=executor_class,
                env=env,
                game=game,
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
                evaluation_lookback=lookback,
                allow_self_termination=False,
                score_mode=score_mode,
                guidance=guidance_str or None,
                goal_condition=goal_condition,
                checker_vlm_model=model_name,
                checker_vlm_kind=vlm_kind,
                checker_max_new_tokens=checker_max_new_tokens,
                parameters=parameters,
                vlm_model=model_name,
                vlm_kind=vlm_kind,
            )

            result = supervisor.evaluate()

            if verbose:
                print(f"  Attempt {attempt}: {'success' if result.get('success') else 'failure'} | score={result.get('score', float('nan')):.3f}")

            pkl_name = f"{group_idx}_{attempt}.pkl"
            with open(os.path.join(out_dir, pkl_name), "wb") as f:
                pickle.dump(result["vlm_call_log"], f)

            rows.append({
                "group_idx": group_idx,
                "attempt": attempt,
                "task_string": task_str,
                "success": result.get("success", float("nan")),
                "score": result.get("score", float("nan")),
            })

            with open(checkpoint_path, "w") as f:
                json.dump(rows, f, indent=2)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved results      -> {csv_path}")
    print(f"Saved episode pkls -> {out_dir}/")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
