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
       Reset env to init_state, then run AttemptCheckerSupervisor(
           task          = task_str,
           executor_class = <click option>,
           env           = env for this init_state,
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import click
import numpy as np
from tqdm import tqdm

from gameboy_worlds import get_environment
from execution.registry import AVAILABLE_EXECUTORS
from execution.report import EnvironmentStepRecord
from execution.supervisors import AttemptCheckerSupervisor, derive_critique_hint
from utils import log_info, log_warn, log_error, VLM, HuggingFaceModel


# The critique prompts and the slice-then-consolidate implementation live in
# execution/supervisors/checker.py so every caller derives hints from identical prompts.


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


def _attempt_task(
    group_idx: str,
    init_state: str,
    task_str: str,
    game: str,
    model_name: str,
    vlm_kind: str,
    executor_class,
    max_steps: int,
    max_tool_calls: int,
    lookback: int,
    controller_variant: str,
    max_attempts: int,
    checker_max_new_tokens: int,
    max_new_tokens: int,
    critique_vlm: VLM,
    parameters: dict,
    verbose: bool,
) -> tuple[dict, tuple | None]:
    """Run all attempts for one (init_state, task) pair in its own env, then close it."""
    env = get_environment(
        game=game,
        controller_variant=controller_variant,
        environment_variant="default",
        init_state=init_state,
        max_steps=max_steps,
        headless=True,
        save_video=False,
    )

    hint = ""
    result = None
    trajectory = None

    try:
        for attempt in range(max_attempts):
            env.reset()

            if verbose:
                print(f"  Attempt {attempt + 1}/{max_attempts}" + (f" | hint: {hint}" if hint else ""))

            supervisor = AttemptCheckerSupervisor(
                task=task_str,
                executor_class=executor_class,
                env=env,
                game=game,
                max_steps=max_steps,
                max_tool_calls=max_tool_calls,
                evaluation_lookback=lookback,
                allow_self_termination=True,
                supervisor_vlm_model=model_name,
                supervisor_vlm_kind=vlm_kind,
                max_new_tokens=checker_max_new_tokens,
                parameters=parameters,
                vlm_model=model_name,
                vlm_kind=vlm_kind,
                hint=hint or None,
            )
            result = supervisor.evaluate()
            # evaluate() returns {"report": SupervisorReport, ...the checker's verdict}. The
            # steps live on the report's single executor leg rather than being copied into
            # the verdict dict.
            executor_reports = result["report"].executor_reports
            env_steps = [
                s for report in executor_reports for s in report.steps
                if isinstance(s, EnvironmentStepRecord)
            ]
            trajectory = _reconstruct_trajectory(env_steps, init_state)

            if verbose:
                print(f"  Result: {'success' if result['success'] else 'failure'} | {result.get('description', '')}")

            if result["success"]:
                break

            if attempt < max_attempts - 1:
                hint = derive_critique_hint(
                    env_steps, task_str, game, critique_vlm, max_new_tokens, hint
                )
                if verbose:
                    print(f"  Derived hint: {hint}")
    finally:
        env.close()

    result_record = {
        "init_state": init_state,
        "task_string": task_str,
        "success": result["success"],
        "description": result.get("description", ""),
        "reasoning": result.get("reasoning", ""),
        "n_tries": attempt + 1,
        # The hint in force on the final attempt. n_tries alone says a hint was derived but
        # not what it said, and this is the information the retained trajectory was actually
        # produced under.
        "final_hint": hint,
        # How the final attempt's executor stopped, and how much budget it left. With
        # allow_self_termination on, "agent_done" is the executor's own claim to have
        # finished; `success` above is the checker judging the same trajectory
        # independently. Keeping both is what lets `debug.py attempt` report how often the
        # completion check is wrong — the failure that silently truncates a trajectory and
        # then ships it forward as a success.
        "termination_reason": result.get("termination_reason"),
        "n_env_steps": result.get("n_env_steps"),
        "max_steps": result.get("max_steps"),
    }
    return result_record, trajectory


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
    default=2000,
    show_default=True,
    help="Token budget for each checker VLM call.",
)
@click.option(
    "--max_concurrency",
    default=16,
    show_default=True,
    help="Max concurrent (init_state, task) attempt pipelines. Forced to 1 for --verbose or a huggingface vlm_kind.",
)
@click.pass_obj
def attempt_tasks_cmd(
    obj, tasks_path, executor_name, max_steps, max_tool_calls, lookback, controller_variant, max_attempts, checker_max_new_tokens, max_concurrency
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

    if os.path.exists(csv_path) and not overwrite:
        log_info(f"Skipping attempts — output already exists at {csv_path}. Use --overwrite to rerun.")
        return

    if os.path.exists(checkpoint_json) and not overwrite:
        with open(checkpoint_json, "r") as f:
            results = json.load(f)
        # An absent pkl means every result is unpaired, which the reconciliation below
        # turns into a full recompute. Reading it unconditionally would instead be a
        # FileNotFoundError on a resume where the small json write landed and the large
        # pkl write did not (SIGTERM between the two, or a /project2 quota failure).
        trajectories = {}
        if os.path.exists(checkpoint_pkl):
            with open(checkpoint_pkl, "rb") as f:
                trajectories = pickle.load(f)
        # A group is "done" only when BOTH its result and its trajectory are on disk.
        # The two checkpoints are separate non-atomic writes (json first), so a run
        # killed between them resumes with a result whose trajectory never landed. The
        # skip check below keys off `results` alone, so without this that group would be
        # skipped forever and its frames lost silently — and since
        # success_trajectories.{json,pkl} are derived from these two different dicts, the
        # json would claim a success the pkl has no trajectory for. Dropping the unpaired
        # results makes the pair the unit of truth and the mismatch self-healing.
        unpaired = [group_idx for group_idx in results if group_idx not in trajectories]
        for group_idx in unpaired:
            del results[group_idx]
        if unpaired:
            log_warn(
                f"checkpoint mismatch: {len(unpaired)} group_idx(s) had a saved result but "
                f"no saved trajectory ({unpaired[:5]}{' ...' if len(unpaired) > 5 else ''}) "
                "— rerunning them."
            )
        log_info(
            f"Resuming from checkpoint — {len(results)} group_idxs already done.",
            parameters,
        )
    else:
        results = {}
        trajectories = {}

    with open(tasks_path, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    jobs = []
    for line_number, record in enumerate(lines):
        init_state = record["init_state"]
        tasks = record.get("tasks", [])
        for task_index, task_str in enumerate(tasks):
            group_idx = f"{line_number}_{task_index}"
            if group_idx in results and not overwrite:
                continue
            jobs.append((group_idx, init_state, task_str))

    # (init_state, task) attempt pipelines each create and close their own env,
    # so they're independent and can run concurrently. --verbose runs interleaved
    # print debugging and HuggingFaceModel isn't safe for concurrent generate()
    # calls — both fall back to max_workers=1, which processes jobs one at a time
    # in submission order (i.e. identical to the old sequential loop).
    effective_workers = (
        1 if (verbose or isinstance(critique_vlm._vlm, HuggingFaceModel)) else max_concurrency
    )

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_to_group = {
            executor.submit(
                _attempt_task,
                group_idx,
                init_state,
                task_str,
                game,
                model_name,
                vlm_kind,
                executor_class,
                max_steps,
                max_tool_calls,
                lookback,
                controller_variant,
                max_attempts,
                checker_max_new_tokens,
                max_new_tokens,
                critique_vlm,
                parameters,
                verbose,
            ): group_idx
            for group_idx, init_state, task_str in jobs
        }

        for future in tqdm(as_completed(future_to_group), total=len(future_to_group), desc="Attempting tasks"):
            group_idx = future_to_group[future]
            result_record, trajectory = future.result()
            results[group_idx] = result_record
            trajectories[group_idx] = trajectory

            with open(checkpoint_json, "w") as f:
                json.dump(results, f, indent=2)
            with open(checkpoint_pkl, "wb") as f:
                pickle.dump(trajectories, f)

    # Write final outputs. all_trajectories.csv goes first and unconditionally: it is the
    # record of what was attempted and why it failed, and the debug tooling reads it.
    # description/reasoning/final_hint are carried through to the CSV because the only other
    # copy lives in checkpoint.json, which is deleted a few lines below on a successful run.
    # They are the judge's stated rationale and the hint the final attempt ran under — the
    # evidence for why a task failed, or why a "success" should be believed.
    pd.DataFrame([
        {
            "group_idx": group_idx,
            "init_state": res["init_state"],
            "task_string": res["task_string"],
            "success": res["success"],
            "n_tries": res["n_tries"],
            # TODO(legacy-cols): the .get defaults exist only so a run resuming from a
            # checkpoint.json written before 2026-07-20 (which has no final_hint key) does
            # not KeyError. Once every in-flight job has finished and no pre-2026-07-20
            # checkpoint remains on disk, index these directly.
            "final_hint": res.get("final_hint", ""),
            "judge_description": res.get("description", ""),
            "judge_reasoning": res.get("reasoning", ""),
            # .get for the same legacy-checkpoint reason as final_hint above: a resumed run
            # can hold records written before these keys existed.
            "termination_reason": res.get("termination_reason"),
            "n_env_steps": res.get("n_env_steps"),
            "max_steps": res.get("max_steps"),
        }
        for group_idx, res in results.items()
    ]).to_csv(csv_path, index=False)

    success_trajectories = {gid: traj for gid, traj in trajectories.items() if results.get(gid, {}).get("success")}
    success_json = {gid: res["task_string"] for gid, res in results.items() if res.get("success")}

    # Fail here rather than let an empty success set propagate. build_info keys off this
    # stem, and an empty one yields an empty info document with no error. The success files
    # are NOT written, so build_info fails loudly on a missing input rather than silently
    # distilling nothing; the checkpoints are left for inspection.
    if not success_json:
        log_error(
            f"attempt_tasks: 0 of {len(results)} attempted tasks succeeded, so there are no "
            f"trajectories to distil an info document from. Attempt records were still written to "
            f"{csv_path} — inspect them before rerunning. Rerunning this stage needs "
            "--overwrite, since that CSV is its skip-if-exists marker.",
            parameters,
        )

    with open(pkl_path, "wb") as f:
        pickle.dump(success_trajectories, f)
    with open(json_path, "w") as f:
        json.dump(success_json, f, indent=2)

    log_info(f"Saved results       -> {csv_path}")
    log_info(f"Saved success tasks -> {json_path}")
    log_info(f"Saved trajectories  -> {pkl_path}")

    if os.path.exists(checkpoint_json):
        os.remove(checkpoint_json)
    if os.path.exists(checkpoint_pkl):
        os.remove(checkpoint_pkl)
