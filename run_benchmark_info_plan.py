"""
Called by scripts/benchmark_info_plan.sh. Use --help for CLI options.

The plan arm's benchmark runner: a sibling of run_benchmark_info.py that constructs an
InfoPlanSupervisor instead of an InfoHintSupervisor. Both read the same info documents and
select from them with the same relevance pass; they differ in what they do with what they
found. run_benchmark_info.py spends it on one hint written at the opening frame and then
lets the executor run unattended. This spends it on a plan, then stays in the loop —
running each step under its own executor call, judging from the frames whether the step
actually landed, hinting when it did not, and rewriting the step when hinting keeps
failing.

Rows carry run_benchmark.py's columns, so viz.py and the debug comparators read this arm
like any other, plus the plan bookkeeping the arm exists to produce. The `hint` column
holds the whole plan, [STEP]-joined, so it lines up with the hint the info arm records.

Budgets: --max_leg_steps caps a single executor call and is internal to the supervisor;
--max_steps caps the emulator steps across the whole episode, retries included, so this arm
and the info arm play with the same allowance and their CSVs stay comparable.

This costs materially more VLM calls per episode than the info arm — a relevance pass, a
plan, then per attempt a windowed judgement plus a hint, plus a revision every few
failures. Budget accordingly before sweeping a whole game.
"""

import json
import os
import traceback

import click
import pandas as pd
from tqdm import tqdm

from gameboy_worlds import AVAILABLE_GAMES, get_benchmark_tasks, get_test_environment

from execution.registry import AVAILABLE_EXECUTORS
from execution.supervisors import PLAN_SEPARATOR, InfoPlanSupervisor
from run_benchmark_info import _load_documents, _load_insight_rows
from utils import load_parameters, log_error, log_info


def _join_leg_reports(supervisor):
    """One `report` cell covering every attempt, under greppable headers.

    Storing only the final attempt would hide the retries, which in this arm are the
    interesting part: the same step attempted three times under three hints is the record of
    what the supervisor tried and how the executor responded to each.
    """
    if not supervisor.leg_reports:
        return None
    blocks = []
    for leg in supervisor.leg_reports:
        step = leg["step"] if leg["step"] is not None else "(unplanned)"
        header = (f"===== STEP {leg['step_index'] + 1} ATTEMPT {leg['attempt']} "
                  f"[{leg['report'].termination_reason}] {step} =====")
        if leg["hint"] and leg["hint"] != leg["step"]:
            header += f"\nHINT: {leg['hint']}"
        blocks.append(f"{header}\n{leg['report']}")
    return "\n\n".join(blocks)


def _plan_summary(supervisor):
    """Flat counts for the CSV, derived from step_log and the supervisor's own counters."""
    attempts = [a for record in supervisor.step_log for a in record["attempts"]]
    return {
        "n_plan_steps": len(supervisor.plan),
        "n_steps_cleared": sum(1 for r in supervisor.step_log if r["cleared"]),
        "n_attempts": len(attempts),
        "n_replans": sum(len(r["replans"]) for r in supervisor.step_log),
        # Without these an over-aggressive filter and a bad planner are indistinguishable
        # from the outcome alone.
        "n_insights_candidate": supervisor.n_insights_candidate,
        "n_insights_kept": supervisor.n_insights_kept,
        "n_insights_distilled": supervisor.n_insights_distilled,
        "n_supervisor_calls": len(supervisor.supervisor_calls),
    }


def run_task(row, max_resets, controller_variant, executor_class, max_tool_calls,
             documents, insight_rows, mode, plan_vlm_model, plan_vlm_kind, max_concurrency,
             max_leg_steps, max_attempts_per_step, max_replans, max_frames_per_slice,
             plan_max_new_tokens, hint_max_new_tokens, judge_max_new_tokens, parameters,
             vlm_model=None, vlm_kind=None, verbose=False, **emulator_kwargs):
    """One benchmark task: plan from the document, then supervise the plan to completion."""
    success = False
    n_resets = 1
    n_steps_total = 0
    n_invalid_total = 0
    subgoals_reached = []
    subgoals_all = None
    mission = row["task"]
    task_str = mission.replace(" ", "_").lower()
    emulator_kwargs = emulator_kwargs.copy()
    emulator_kwargs["session_name"] += f"/{task_str}/"
    emulator_kwargs["wait_ticks"] = 20
    error = True
    report_str = None
    plan_text = None
    selected_ids = []
    summary = {"n_plan_steps": 0, "n_steps_cleared": 0, "n_attempts": 0, "n_replans": 0,
               "n_insights_candidate": 0, "n_insights_kept": 0, "n_insights_distilled": 0,
               "n_supervisor_calls": 0}
    step_log = []
    supervisor_calls = []
    insights_block = None
    try:
        while n_resets < max_resets + 1:
            environment = get_test_environment(
                row=row, controller_variant=controller_variant, **emulator_kwargs
            )
            supervisor = InfoPlanSupervisor(
                task=mission,
                executor_class=executor_class,
                env=environment,
                game=row["game"],
                max_steps=emulator_kwargs["max_steps"],
                max_tool_calls=max_tool_calls,
                documents=documents,
                insight_rows=insight_rows,
                mode=mode,
                init_state=row["init_state"],
                hint_vlm_model=plan_vlm_model,
                hint_vlm_kind=plan_vlm_kind,
                max_concurrency=max_concurrency,
                max_leg_steps=max_leg_steps,
                max_attempts_per_step=max_attempts_per_step,
                max_replans=max_replans,
                max_frames_per_slice=max_frames_per_slice,
                plan_max_new_tokens=plan_max_new_tokens,
                hint_max_new_tokens=hint_max_new_tokens,
                judge_max_new_tokens=judge_max_new_tokens,
                verbose=verbose,
                parameters=parameters,
                vlm_model=vlm_model,
                vlm_kind=vlm_kind,
            )
            report = supervisor.evaluate()
            plan_text = PLAN_SEPARATOR.join(supervisor.plan) if supervisor.plan else None
            selected_ids = list(supervisor.selected_ids)
            summary = _plan_summary(supervisor)
            # step_log holds the supervisor's decisions; supervisor_calls holds the prompts
            # and replies behind them. Strip nothing from either — they are the artefact
            # this arm exists to produce, and the only record of the supervisor's own
            # reasoning, which never reaches an ExecutorReport.
            step_log = supervisor.step_log
            supervisor_calls = list(supervisor.supervisor_calls)
            insights_block = supervisor.insights_block

            last_state = environment.get_info()
            if subgoals_all is None:
                subgoals_all = last_state["subgoals"]["all"]
            for subgoal in last_state["subgoals"]["completed"]:
                if subgoal not in subgoals_reached:
                    subgoals_reached.append(subgoal)

            n_steps_total += last_state["core"]["steps"]
            n_invalid_total += sum(len(leg["report"].invalid_steps)
                                   for leg in supervisor.leg_reports)

            report_str = _join_leg_reports(supervisor)
            if verbose:
                print(f"\n----- trajectory (attempt {n_resets}) " + "-" * 44)
                print(report_str)

            environment.close()
            error = False
            if report is not None and report.termination_reason == "terminated":
                success = True
                break
            n_resets += 1

    except Exception as e:
        error = True
        print(f"Error during execution of task '{mission}': {e}")
        traceback.print_exc()
    return (success, n_resets - 1, n_steps_total, n_invalid_total, subgoals_reached,
            subgoals_all, error, report_str, plan_text, selected_ids, summary, step_log,
            supervisor_calls, insights_block)


@click.command()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--info_docs", default=None, type=str,
              help="Comma-separated info.json path(s). Required for --mode retrieval.")
@click.option("--insights_paths", default=None, type=str,
              help="Comma-separated insights.jsonl path(s). Required for --mode init_state.")
@click.option("--mode", default="retrieval", type=click.Choice(["retrieval", "init_state"]),
              help="How the planner's candidate knowledge is selected. Identical to "
                   "run_benchmark_info.py's --mode; the selection code is shared.")
@click.option("--plan_vlm_model", default=None, type=str,
              help="Model for the selection, planning, judging, hinting and revision calls. "
                   "Defaults to the executor's.")
@click.option("--plan_vlm_kind", default=None, type=str)
@click.option("--max_concurrency", default=8, show_default=True, type=int,
              help="Parallel relevance calls during selection; they are independent.")
@click.option("--max_leg_steps", default=5, show_default=True, type=int,
              help="Env-step cap for ONE executor attempt. Internal to the supervisor: it "
                   "decides how often the supervisor gets to look, not the episode budget.")
@click.option("--max_attempts_per_step", default=3, show_default=True, type=int,
              help="Failed attempts at one step before the supervisor moves on. With per-step "
                   "rewriting removed this is the only exit from the retry loop besides "
                   "the episode budget. The final step ignores it.")
@click.option("--max_replans", default=2, show_default=True, type=int,
              help="Times the plan may be rewritten in one episode. Bounds both cost and "
                   "the risk of thrashing between two readings of the same screen.")
@click.option("--max_frames_per_slice", default=8, show_default=True, type=int,
              help="Trajectory frames per judging call.")
@click.option("--plan_max_new_tokens", default=5000, show_default=True, type=int,
              help="Token budget for the planning, revision and segment-summary calls. A "
                   "plan is several sentences per step, so this is the one that truncates "
                   "first — and a truncated plan loses its trailing steps silently.")
@click.option("--hint_max_new_tokens", default=2400, show_default=True, type=int,
              help="Token budget for the relevance, judging and hint calls. Each answers in "
                   "a couple of lines, but a model that reasons before answering needs room "
                   "to reach the line the parser looks for.")
@click.option("--judge_max_new_tokens", default=4800, show_default=True, type=int,
              help="Token budget for the completion check. Largest of the three: the "
                   "Complete: verdict is the last line of the reply, so a truncated "
                   "response loses the answer and reads as not complete, and every step "
                   "then fails its judgement regardless of the screen.")
@click.option("--executor_max_new_tokens", default=8000, show_default=True, type=int,
              help="Token budget per executor action call. Overrides the project-wide "
                   "executor_vlm_max_new_tokens for this process only, so raising it here "
                   "does not change what the other arms' runs were scored under.")
@click.option("--controller_variant", default="low_level", type=str)
@click.option("--executor", default="simple", type=click.Choice(list(AVAILABLE_EXECUTORS.keys())))
@click.option("--executor_vlm_model", default=None, type=str)
@click.option("--executor_vlm_kind", default=None, type=str)
@click.option("--save_video", type=bool, default=True)
@click.option("--max_resets", default=1, type=int)
@click.option("--max_steps", default=200, type=int,
              help="Emulator steps for the WHOLE episode, every attempt and retry included.")
@click.option("--max_tool_calls", default=0, type=int)
@click.option("--override_index", default=None, type=int, required=False)
@click.option("--random_sample", type=int, default=None,
              help="Run a RANDOM sample of n tasks (seed 42, shared with "
                   "run_benchmark_info.py so both arms draw the same ones).")
@click.option("--n_tasks", type=int, default=None,
              help="Run the FIRST n tasks in benchmark order. Use this, not --random_sample, "
                   "for a quick look at a fixed prefix; the two cannot be combined.")
@click.option("--verbose", is_flag=True, default=False,
              help="Print the supervisor's per-attempt narration (step, hint, judge verdict, "
                   "revisions) and the full trajectory after each episode. Without it the "
                   "run prints only the tqdm bar and one summary line per episode.")
@click.option("--regenerate", is_flag=True, default=False)
def do(game, info_docs, insights_paths, mode, plan_vlm_model, plan_vlm_kind, max_concurrency,
       max_leg_steps, max_attempts_per_step, max_replans, max_frames_per_slice,
       plan_max_new_tokens, hint_max_new_tokens, judge_max_new_tokens,
       executor_max_new_tokens,
       controller_variant, executor, executor_vlm_model, executor_vlm_kind, save_video,
       max_resets, max_steps, max_tool_calls, override_index, random_sample, n_tasks, verbose,
       regenerate):
    project_parameters = load_parameters()
    vlm_name = executor_vlm_model or project_parameters["executor_vlm_model"]
    model_save_name = vlm_name.split("/")[-1].lower()

    # In-process only: the loaded dict is handed to every supervisor and executor below, and
    # nothing writes it back to configs/, so the other arms keep the project-wide value.
    previous = project_parameters.get("executor_vlm_max_new_tokens")
    project_parameters["executor_vlm_max_new_tokens"] = executor_max_new_tokens
    log_info(f"Executor token budget: {previous} -> {executor_max_new_tokens} "
             f"(this run only). Plan calls: {plan_max_new_tokens}, "
             f"judge: {judge_max_new_tokens}, hint: {hint_max_new_tokens}.")

    documents, insight_rows = None, None
    if mode == "retrieval":
        if not info_docs:
            log_error("--mode retrieval requires --info_docs.", project_parameters)
        documents = _load_documents(info_docs, project_parameters)
    else:
        if not insights_paths:
            log_error("--mode init_state requires --insights_paths.", project_parameters)
        insight_rows = _load_insight_rows(insights_paths, project_parameters)

    session_name = f"benchmark_info_plan_{mode}_{executor}_{model_save_name}"
    emulator_kwargs = {
        "headless": True,
        "save_video": save_video,
        "session_name": session_name,
        "max_steps": max_steps,
    }
    benchmark_tasks = get_benchmark_tasks(game=game)
    columns = [
        "game", "task", "success", "n_resets", "n_steps", "n_invalid",
        "subgoals_reached", "all_subgoals", "report",
        # `hint` holds the [STEP]-joined plan, matching run_benchmark_info.py's column so the
        # two arms' advice sits in the same place.
        "hint",
        "n_plan_steps", "n_steps_cleared", "n_attempts", "n_replans",
        "n_insights_candidate", "n_insights_kept", "n_insights_distilled",
        "n_supervisor_calls",
        "selected_entry_ids", "insights_block", "step_log",
        # Every prompt and reply the supervisor produced. Large, and deliberately so: it is
        # the only place the filter, the distillation, the plan and the judgements can be
        # read back after a run.
        "supervisor_calls",
    ]
    if random_sample is not None and n_tasks is not None:
        raise ValueError("--random_sample and --n_tasks select the task set in different "
                         "ways; pass one or the other.")
    if random_sample is not None:
        if not (1 <= random_sample <= len(benchmark_tasks) - 1):
            raise ValueError(
                f"random_sample must be between 1 and {len(benchmark_tasks) - 1}, got {random_sample}"
            )
        # Same seed as run_benchmark_info.py so a sampled run picks the same tasks in both arms.
        benchmark_tasks = benchmark_tasks.sample(n=random_sample, random_state=42).reset_index(drop=True)
    if n_tasks is not None:
        if not (1 <= n_tasks <= len(benchmark_tasks)):
            raise ValueError(
                f"n_tasks must be between 1 and {len(benchmark_tasks)}, got {n_tasks}"
            )
        benchmark_tasks = benchmark_tasks.head(n_tasks).reset_index(drop=True)

    results_dir = project_parameters["results_dir"]
    os.makedirs(f"{results_dir}/benchmark/{game}/", exist_ok=True)
    # A subset run gets its own CSV: resuming a 35-task run from a 5-task file would treat
    # the first 5 as done and silently skip them.
    stem = f"info_plan_{mode}_{executor}_{model_save_name}"
    if random_sample is not None:
        stem += f"_sample{random_sample}"
    if n_tasks is not None:
        stem += f"_first{n_tasks}"
    save_path = f"{results_dir}/benchmark/{game}/{stem}.csv"

    if not regenerate and os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        results = existing_df.values.tolist()
        n_completed = len(existing_df)
        print(f"Resuming from checkpoint: {n_completed} tasks already completed in {save_path}")
    else:
        results = []
        n_completed = 0

    n_planned = 0
    n_run = 0
    for i, row in tqdm(benchmark_tasks.iterrows(), total=len(benchmark_tasks)):
        if i < n_completed:
            continue
        if override_index is not None and i != override_index:
            continue
        if override_index is not None:
            print(f"Running override index {override_index} on row:")
            for column in row.index:
                print(f"  {column}: {row[column]}")

        (success, n_resets, n_steps, n_invalid, subgoals_reached, subgoals_all, error,
         report_str, plan_text, selected_ids, summary, step_log,
         supervisor_calls, insights_block) = run_task(
            row=row,
            max_resets=max_resets,
            controller_variant=controller_variant,
            executor_class=AVAILABLE_EXECUTORS[executor],
            max_tool_calls=max_tool_calls,
            documents=documents,
            insight_rows=insight_rows,
            mode=mode,
            plan_vlm_model=plan_vlm_model or executor_vlm_model,
            plan_vlm_kind=plan_vlm_kind or executor_vlm_kind,
            max_concurrency=max_concurrency,
            max_leg_steps=max_leg_steps,
            max_attempts_per_step=max_attempts_per_step,
            max_replans=max_replans,
            max_frames_per_slice=max_frames_per_slice,
            plan_max_new_tokens=plan_max_new_tokens,
            hint_max_new_tokens=hint_max_new_tokens,
            judge_max_new_tokens=judge_max_new_tokens,
            parameters=project_parameters,
            vlm_model=executor_vlm_model,
            vlm_kind=executor_vlm_kind,
            verbose=verbose,
            **emulator_kwargs,
        )
        if error:
            log_error(f"Error occurred during execution of task '{row['task']}' - exiting loop")
        if plan_text:
            n_planned += 1
        n_run += 1

        print(f"  -> success={success}  steps={n_steps}  "
              f"plan={summary['n_steps_cleared']}/{summary['n_plan_steps']} steps cleared over "
              f"{summary['n_attempts']} attempt(s), {summary['n_replans']} replan(s)  "
              f"insights={summary['n_insights_kept']}/{summary['n_insights_candidate']}"
              f"->{summary['n_insights_distilled']}  "
              f"{summary['n_supervisor_calls']} supervisor calls")

        results.append([
            row["game"], row["task"], success, n_resets, n_steps, n_invalid,
            subgoals_reached, subgoals_all, report_str, plan_text,
            summary["n_plan_steps"], summary["n_steps_cleared"], summary["n_attempts"],
            summary["n_replans"],
            summary["n_insights_candidate"], summary["n_insights_kept"],
            summary["n_insights_distilled"], summary["n_supervisor_calls"],
            json.dumps(selected_ids),
            insights_block,
            # default=str so a report object or anything else non-serialisable that finds its
            # way into step_log degrades to text instead of losing the whole row.
            json.dumps(step_log, default=str),
            json.dumps(supervisor_calls, default=str),
        ])
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(save_path, index=False)
        log_info(f"Saved benchmark results to {save_path}")

    # An episode that produced no plan ran as an unhinted baseline, so a low number here
    # means the planner is failing, not that planning does not help.
    log_info(f"Produced a plan on {n_planned}/{max(n_run, 1)} episodes this run "
             f"({100.0 * n_planned / max(n_run, 1):.0f}%).")


if __name__ == "__main__":
    do()
