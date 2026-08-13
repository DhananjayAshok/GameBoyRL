"""
The plan arm: plan from the info document, then supervise the executor through the plan
step by step.

The episode runs each plan step under its own executor call, judges from the frames whether
the step landed, hints when it did not, and rewrites the plan when hinting keeps failing.

``--mode`` decides only where the planner's knowledge comes from. Everything after that —
the relevance pass, the insight filter, the planner, the step loop — is identical, which is
what makes the two modes comparable:

``retrieval``
    Documents distilled from real trajectories by ``vlm_scripts/build_info.py``, named with
    ``--info_docs``.

``parametric``
    One document written by the model from its own knowledge of the game, given nothing but
    the game's name. Generated on first use and cached at ``Paths.parametric_doc()``.

The pair is the control the arm was missing. "Planning from a distilled document beats the
baseline" does not distinguish *distillation worked* from *any game-specific text worked*;
retrieval-vs-parametric does, and where they score alike the build pipeline has not earned
its cost.

Budgets: ``--max_leg_steps`` caps a single executor call and is internal to the supervisor;
``--max_steps`` caps emulator steps across the whole episode, retries included, so this arm
and the baseline play with the same allowance and their CSVs stay comparable.

This costs materially more VLM calls per episode than the baseline — a relevance pass, a
plan, then per attempt a windowed judgement plus a hint, plus a revision every few failures.
Budget accordingly before sweeping a whole game.
"""

from __future__ import annotations

import json

import click

from gameboy_worlds import get_benchmark_tasks

from execution.parametric_doc import load_or_generate_parametric_document
from execution.registry import AVAILABLE_EXECUTORS
from execution.supervisors import PLAN_SEPARATOR, InfoSubgoalSupervisor
from utils import VLM, log_error, log_info
from utils.paths import Paths

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
        # `n_plan_steps` is the CURRENT plan; `n_slots_attempted` is how many target slots
        # the loop worked through, and is the denominator `n_steps_cleared` belongs over.
        # They differ after a replan. `n_original_plan_steps` is the plan as first written.
        "planned": result["planned"],
        "n_plan_steps": len(result["plan"]),
        "n_original_plan_steps": len(result["original_plan"]),
        "n_slots_attempted": len(step_log),
        "n_steps_cleared": sum(1 for r in step_log if r["cleared"]),
        "n_attempts": len(attempts),
        "n_replans": result["n_replans"],
        # Without these an over-aggressive filter and a bad planner are indistinguishable
        # from the outcome alone.
        "n_insights_candidate": result["n_insights_candidate"],
        "n_insights_kept": result["n_insights_kept"],
        "n_insights_distilled": result["n_insights_distilled"],
        "n_supervisor_calls": len(report.supervisor_calls),
    }


SUMMARY_COLUMNS = [
    "planned", "n_plan_steps", "n_original_plan_steps", "n_slots_attempted",
    "n_steps_cleared", "n_attempts", "n_replans",
    "n_insights_candidate", "n_insights_kept", "n_insights_distilled",
    "n_supervisor_calls",
]

_EMPTY_SUMMARY = {key: 0 for key in SUMMARY_COLUMNS}


@click.command(name="info_subgoal")
@click.option("--info_docs", default=None, type=str,
              help="Comma-separated info.json path(s). Required for --mode retrieval.")
@click.option("--mode", default="retrieval", type=click.Choice(["retrieval", "parametric"]),
              help="Where the planner's knowledge comes from: documents distilled from real "
                   "trajectories (retrieval), or one the model writes from its own priors "
                   "given only the game's name (parametric). Everything downstream is "
                   "identical, so the pair isolates what distillation actually bought.")
@click.option("--parametric_categories", default=10, show_default=True, type=int,
              help="Upper bound on task categories requested when generating a parametric "
                   "document. Ignored under --mode retrieval. The generation call uses "
                   "--supervisor_max_new_tokens like every other call on that model; the "
                   "whole document comes back in one reply, so ask for fewer categories "
                   "rather than lowering that budget.")
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
@click.option("--executor_max_new_tokens", default=8000, show_default=True, type=int,
              help="Token budget per executor action call. Overrides the project-wide "
                   "executor_vlm_max_new_tokens for this process only.")
@click.pass_obj
def info_subgoal_cmd(obj, info_docs, mode, parametric_categories,
                      max_concurrency, max_leg_steps, max_attempts_per_step, max_replans,
                      max_frames_per_slice, executor_max_new_tokens):
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
             f"(this run only). Supervisor calls: {obj['supervisor_max_new_tokens']}.")

    # Resolved on the group (falling back to the executor's), because both the parametric
    # document's cache path and the supervisor's own VLM are keyed on it and must not
    # disagree.
    knowledge_model = obj["supervisor_vlm_model"]
    knowledge_kind = obj["supervisor_vlm_kind"]

    if mode == "retrieval":
        if not info_docs:
            log_error("--mode retrieval requires --info_docs.", parameters)
        documents = common.load_documents(info_docs, parameters)
    else:
        # Generated once and cached, so a rerun of the same command plans from the same
        # document. Written by the supervisor's own model — there is deliberately no
        # separate generator flag — and keyed on it rather than on the executor's: a
        # different model has different priors, and reusing one's document under another's
        # name would attribute knowledge to a model that never wrote it.
        doc_path = Paths(parameters=parameters, game=game,
                         model_name=knowledge_model).parametric_doc()
        documents = [load_or_generate_parametric_document(
            game=game,
            path=doc_path,
            vlm=VLM(knowledge_model, knowledge_kind),
            n_categories=parametric_categories,
            max_new_tokens=obj["supervisor_max_new_tokens"],
            regenerate=obj["regenerate"],
            parameters=parameters,
        )]

    emulator_kwargs = {
        "headless": True,
        "save_video": obj["save_video"],
        "session_name": f"benchmark_info_subgoal_{mode}_{executor}_{model_save_name}",
        "max_steps": obj["max_steps"],
    }

    columns = common.COMMON_COLUMNS + [
        # `hint` holds the [STEP]-joined plan, matching the info arm's column so the two
        # arms' advice sits in the same place. `original_plan` is the same, before replans.
        "hint",
        "original_plan",
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
        parameters, game, f"info_subgoal_{mode}_{executor}_{model_save_name}",
        obj["n_tasks"])
    results, n_completed = common.load_checkpoint(save_path, obj["regenerate"],
                                                  columns, parameters)

    def run_one(row):
        def play(environment):
            supervisor = InfoSubgoalSupervisor(
                task=row["task"],
                executor_class=executor_class,
                env=environment,
                game=row["game"],
                max_steps=obj["max_steps"],
                max_tool_calls=obj["max_tool_calls"],
                documents=documents,
                supervisor_vlm_model=knowledge_model,
                supervisor_vlm_kind=knowledge_kind,
                max_new_tokens=obj["supervisor_max_new_tokens"],
                max_concurrency=max_concurrency,
                max_leg_steps=max_leg_steps,
                max_attempts_per_target=max_attempts_per_step,
                max_replans=max_replans,
                max_frames_per_slice=max_frames_per_slice,
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
                    "original_plan": (PLAN_SEPARATOR.join(result["original_plan"])
                                      if result["original_plan"] else None),
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
            arm="info_subgoal",
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
            extras.get("original_plan"),
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
              f"plan={summary['n_steps_cleared']}/{summary['n_slots_attempted']} steps "
              f"cleared over {summary['n_attempts']} attempt(s), "
              f"{summary['n_replans']} replan(s)"
              f"{'' if summary['planned'] else '  [UNPLANNED]'}  "
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
