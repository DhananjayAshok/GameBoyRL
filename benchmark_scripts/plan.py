"""
The plan arm: plan from the info document, then supervise the executor through the plan
step by step.

A sibling of the hint arm reading the same documents with the same relevance pass; the two
differ in what they do with what they found. The hint arm spends it on one hint written at
the opening frame and then lets the executor run unattended. This spends it on a plan and
stays in the loop — running each step under its own executor call, judging from the frames
whether the step landed, hinting when it did not, and rewriting the step when hinting keeps
failing.

Budgets: ``--max_leg_steps`` caps a single executor call and is internal to the supervisor;
``--max_steps`` caps emulator steps across the whole episode, retries included, so this arm
and the hint arm play with the same allowance and their CSVs stay comparable.

This costs materially more VLM calls per episode than the hint arm — a relevance pass, a
plan, then per attempt a windowed judgement plus a hint, plus a revision every few failures.
Budget accordingly before sweeping a whole game.
"""

from __future__ import annotations

import json

import click

from gameboy_worlds import get_benchmark_tasks

from execution.registry import AVAILABLE_EXECUTORS
from execution.supervisors import PLAN_SEPARATOR, InfoPlanSupervisor
from utils import log_error, log_info

from benchmark_scripts import common


def _plan_summary(result: dict, report) -> dict:
    """Flat counts for the CSV, from the supervisor's returned extras and its report.

    Reads the ``evaluate()`` return value rather than the supervisor object: the return
    value is the contract, so a count here cannot go stale against an attribute that was
    renamed. ``_join_leg_reports`` used to live beside this to stitch the per-attempt
    trajectories into one CSV cell; ``SupervisorReport.__str__`` renders the interleaved
    event log directly, so the stitching is gone.
    """
    step_log = result["step_log"]
    attempts = [a for record in step_log for a in record["attempts"]]
    return {
        "n_plan_steps": len(result["plan"]),
        "n_steps_cleared": sum(1 for r in step_log if r["cleared"]),
        "n_attempts": len(attempts),
        "n_replans": sum(len(r["replans"]) for r in step_log),
        # Without these an over-aggressive filter and a bad planner are indistinguishable
        # from the outcome alone.
        "n_insights_candidate": result["n_insights_candidate"],
        "n_insights_kept": result["n_insights_kept"],
        "n_insights_distilled": result["n_insights_distilled"],
        "n_supervisor_calls": len(report.supervisor_calls),
    }


_EMPTY_SUMMARY = {
    "n_plan_steps": 0, "n_steps_cleared": 0, "n_attempts": 0, "n_replans": 0,
    "n_insights_candidate": 0, "n_insights_kept": 0, "n_insights_distilled": 0,
    "n_supervisor_calls": 0,
}

SUMMARY_COLUMNS = [
    "n_plan_steps", "n_steps_cleared", "n_attempts", "n_replans",
    "n_insights_candidate", "n_insights_kept", "n_insights_distilled",
    "n_supervisor_calls",
]


@click.command(name="plan")
@click.option("--info_docs", default=None, type=str,
              help="Comma-separated info.json path(s). Required for --mode retrieval.")
@click.option("--insights_paths", default=None, type=str,
              help="Comma-separated insights.jsonl path(s). Required for --mode init_state.")
@click.option("--mode", default="retrieval", type=click.Choice(["retrieval", "init_state"]),
              help="How the planner's candidate knowledge is selected. Identical to the "
                   "info arm's --mode; the selection code is shared.")
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
              help="Failed attempts at one step before the supervisor moves on. The final "
                   "step ignores it.")
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
              help="Token budget for the relevance, judging and hint calls.")
@click.option("--judge_max_new_tokens", default=4800, show_default=True, type=int,
              help="Token budget for the completion check. Largest of the three: the "
                   "Complete: verdict is the last line of the reply, so a truncated "
                   "response loses the answer and reads as not complete.")
@click.option("--executor_max_new_tokens", default=8000, show_default=True, type=int,
              help="Token budget per executor action call. Overrides the project-wide "
                   "executor_vlm_max_new_tokens for this process only.")
@click.pass_obj
def plan_cmd(obj, info_docs, insights_paths, mode, plan_vlm_model, plan_vlm_kind,
             max_concurrency, max_leg_steps, max_attempts_per_step, max_replans,
             max_frames_per_slice, plan_max_new_tokens, hint_max_new_tokens,
             judge_max_new_tokens, executor_max_new_tokens):
    """Benchmark with a plan written from the document and supervised step by step."""
    parameters = obj["parameters"]
    game = obj["game"]
    executor = obj["executor"]
    executor_class = AVAILABLE_EXECUTORS[executor]
    model_save_name = obj["model_save_name"]

    # In-process only: the loaded dict is handed to every supervisor and executor below, and
    # nothing writes it back to configs/, so the other arms keep the project-wide value.
    previous = parameters.get("executor_vlm_max_new_tokens")
    parameters["executor_vlm_max_new_tokens"] = executor_max_new_tokens
    log_info(f"Executor token budget: {previous} -> {executor_max_new_tokens} "
             f"(this run only). Plan calls: {plan_max_new_tokens}, "
             f"judge: {judge_max_new_tokens}, hint: {hint_max_new_tokens}.")

    documents, insight_rows = None, None
    if mode == "retrieval":
        if not info_docs:
            log_error("--mode retrieval requires --info_docs.", parameters)
        documents = common.load_documents(info_docs, parameters)
    else:
        if not insights_paths:
            log_error("--mode init_state requires --insights_paths.", parameters)
        insight_rows = common.load_insight_rows(insights_paths, parameters)

    emulator_kwargs = {
        "headless": True,
        "save_video": obj["save_video"],
        "session_name": f"benchmark_info_plan_{mode}_{executor}_{model_save_name}",
        "max_steps": obj["max_steps"],
    }

    columns = common.COMMON_COLUMNS + [
        # `hint` holds the [STEP]-joined plan, matching the info arm's column so the two
        # arms' advice sits in the same place.
        "hint",
        *SUMMARY_COLUMNS,
        "selected_entry_ids", "insights_block", "step_log",
        # The supervisor's prompts and replies are NOT a column any more. They live on the
        # archived SupervisorReport's event_log, interleaved with the executor legs they
        # drove — which is both where the frames are and where the ordering is meaningful.
        # As a JSON cell they were megabytes of text per row with numpy arrays degraded to
        # str() by `default=str`.
        common.SESSION_COLUMN,
    ]
    tasks = common.select_tasks(get_benchmark_tasks(game=game), obj["n_tasks"])
    save_path = common.results_path(
        parameters, game, f"info_plan_{mode}_{executor}_{model_save_name}", obj["n_tasks"])
    results, n_completed = common.load_checkpoint(save_path, obj["regenerate"],
                                                  columns, parameters)

    def run_one(row):
        def play(environment):
            supervisor = InfoPlanSupervisor(
                task=row["task"],
                executor_class=executor_class,
                env=environment,
                game=row["game"],
                max_steps=obj["max_steps"],
                max_tool_calls=obj["max_tool_calls"],
                documents=documents,
                insight_rows=insight_rows,
                mode=mode,
                init_state=row["init_state"],
                hint_vlm_model=plan_vlm_model or obj["executor_vlm_model"],
                hint_vlm_kind=plan_vlm_kind or obj["executor_vlm_kind"],
                max_concurrency=max_concurrency,
                max_leg_steps=max_leg_steps,
                max_attempts_per_step=max_attempts_per_step,
                max_replans=max_replans,
                max_frames_per_slice=max_frames_per_slice,
                plan_max_new_tokens=plan_max_new_tokens,
                hint_max_new_tokens=hint_max_new_tokens,
                judge_max_new_tokens=judge_max_new_tokens,
                verbose=obj["verbose"],
                parameters=parameters,
                vlm_model=obj["executor_vlm_model"],
                vlm_kind=obj["executor_vlm_kind"],
            )
            result = supervisor.evaluate()
            report = result["report"]
            if obj["verbose"]:
                print("\n----- trajectory " + "-" * 44)
                print(str(report))
            return common.PlayResult(
                report=report,
                extras={
                    "plan": PLAN_SEPARATOR.join(result["plan"]) if result["plan"] else None,
                    "selected_ids": result["selected_ids"],
                    "summary": _plan_summary(result, report),
                    # The supervisor's decisions. The prompts and replies behind them are on
                    # the report's event_log, interleaved with the executor legs they drove,
                    # so they are no longer carried separately.
                    "step_log": result["step_log"],
                    "insights_block": result["insights_block"],
                },
            )

        return common.run_episode(
            row, play,
            arm="plan",
            controller_variant=obj["controller_variant"],
            executor_name=executor_class.__name__,
            model=model_save_name,
            **emulator_kwargs,
        )

    def build_row(row, outcome):
        extras = outcome.extras
        summary = extras.get("summary", _EMPTY_SUMMARY)
        return common.common_row(row, outcome) + [
            extras.get("plan"),
            *[summary[key] for key in SUMMARY_COLUMNS],
            json.dumps(extras.get("selected_ids", [])),
            extras.get("insights_block"),
            # default=str so a report object or anything else non-serialisable that finds
            # its way into step_log degrades to text instead of losing the whole row.
            json.dumps(extras.get("step_log", []), default=str),
            json.dumps(outcome.session_dirs),
        ]

    n_planned = 0
    n_run = 0

    def on_episode(row, outcome):
        nonlocal n_planned, n_run
        summary = outcome.extras.get("summary", _EMPTY_SUMMARY)
        if outcome.extras.get("plan"):
            n_planned += 1
        n_run += 1
        print(f"  -> success={outcome.success}  steps={outcome.n_steps}  "
              f"plan={summary['n_steps_cleared']}/{summary['n_plan_steps']} steps cleared "
              f"over {summary['n_attempts']} attempt(s), {summary['n_replans']} replan(s)  "
              f"insights={summary['n_insights_kept']}/{summary['n_insights_candidate']}"
              f"->{summary['n_insights_distilled']}  "
              f"{summary['n_supervisor_calls']} supervisor calls")

    common.run_sweep(
        tasks,
        columns=columns,
        save_path=save_path,
        results=results,
        n_completed=n_completed,
        override_index=obj["override_index"],
        run_one=run_one,
        build_row=build_row,
        on_episode=on_episode,
    )

    # An episode that produced no plan ran as an unhinted baseline, so a low number here
    # means the planner is failing, not that planning does not help.
    log_info(f"Produced a plan on {n_planned}/{max(n_run, 1)} episodes this run "
             f"({100.0 * n_planned / max(n_run, 1):.0f}%).")
