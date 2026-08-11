"""
The hint arm: read a prebuilt info document, write one hint for the opening screen, then
run the executor with it.

The test-time half of the context-engineering vertical. Two modes decide *which* knowledge
reaches the hint writer; the synthesis call is identical in both, so a difference in results
is attributable to selection alone:

``retrieval``
    Ask, per document entry, whether it fits this task and this screen. Needs --info_docs.

``init_state``
    Skip the document and key off the episode's init state, which the benchmark row gives
    outright — the retrieval-free upper bound. Needs --insights_paths.

Rows carry the baseline's columns plus the hint and its provenance, so viz.py and the debug
comparators read this arm like any other.
"""

from __future__ import annotations

import json

import click

from gameboy_worlds import get_benchmark_tasks

from execution.registry import AVAILABLE_EXECUTORS
from execution.supervisors import InfoHintSupervisor
from utils import log_error, log_info

from benchmark_scripts import common


@click.command(name="info")
@click.option("--info_docs", default=None, type=str,
              help="Comma-separated info.json path(s). Entries from all of them are unioned "
                   "and tagged with a source label. Required for --mode retrieval.")
@click.option("--insights_paths", default=None, type=str,
              help="Comma-separated insights.jsonl path(s). Required for --mode init_state.")
@click.option("--mode", default="retrieval", type=click.Choice(["retrieval", "init_state"]),
              help="How the hint's candidate knowledge is selected. 'init_state' skips the "
                   "document and keys off the episode's init state.")
@click.option("--hint_vlm_model", default=None, type=str,
              help="Model for the relevance and hint-writing calls. Defaults to the executor's.")
@click.option("--hint_vlm_kind", default=None, type=str)
@click.option("--max_concurrency", default=8, show_default=True, type=int,
              help="Parallel relevance calls per episode; they are independent.")
@click.pass_obj
def info_cmd(obj, info_docs, insights_paths, mode, hint_vlm_model, hint_vlm_kind,
             max_concurrency):
    """Benchmark with one hint written from a prebuilt info document."""
    parameters = obj["parameters"]
    game = obj["game"]
    executor = obj["executor"]
    executor_class = AVAILABLE_EXECUTORS[executor]
    model_save_name = obj["model_save_name"]

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
        "session_name": f"benchmark_info_{mode}_{executor}_{model_save_name}",
        "max_steps": obj["max_steps"],
    }

    columns = common.COMMON_COLUMNS + [
        "hint", "n_entries_selected", "selected_entry_ids", common.SESSION_COLUMN,
    ]
    tasks = common.select_tasks(get_benchmark_tasks(game=game), obj["n_tasks"])
    save_path = common.results_path(
        parameters, game, f"info_{mode}_{executor}_{model_save_name}", obj["n_tasks"])
    results, n_completed = common.load_checkpoint(save_path, obj["regenerate"],
                                                  columns, parameters)

    def run_one(row):
        def play(environment):
            supervisor = InfoHintSupervisor(
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
                hint_vlm_model=hint_vlm_model or obj["executor_vlm_model"],
                hint_vlm_kind=hint_vlm_kind or obj["executor_vlm_kind"],
                max_concurrency=max_concurrency,
                vlm_model=obj["executor_vlm_model"],
                vlm_kind=obj["executor_vlm_kind"],
            )
            result = supervisor.evaluate()
            # `selected_ids` says which entries the hint was synthesised from, kept per
            # episode so a hint can be traced back to its evidence from the CSV alone. In
            # init_state mode each id pins the exact stage-A row
            # (source/group_idx#category) in insights.jsonl. Read from the returned dict
            # rather than off the supervisor: the return value is the contract.
            hint, selected_ids = result["hint"], result["selected_ids"]
            if obj["verbose"]:
                print(f"\n  Hint for '{row['task']}': {hint}")
                print(f"  Synthesised from {len(selected_ids)} entr(ies): {selected_ids}")
                for leg in result["report"].executor_reports:
                    leg.show()
            return common.PlayResult(
                report=result["report"],
                extras={"hint": hint, "selected_ids": selected_ids},
            )

        return common.run_episode(
            row, play,
            arm="info",
            controller_variant=obj["controller_variant"],
            executor_name=executor_class.__name__,
            model=model_save_name,
            **emulator_kwargs,
        )

    def build_row(row, outcome):
        selected_ids = outcome.extras.get("selected_ids", [])
        return common.common_row(row, outcome) + [
            outcome.extras.get("hint"),
            len(selected_ids),
            # JSON rather than a Python list repr so it survives the CSV round-trip as
            # something machine-readable: json.loads(row.selected_entry_ids) just works.
            json.dumps(selected_ids),
            json.dumps(outcome.session_dirs),
        ]

    n_hinted = 0

    def on_episode(row, outcome):
        nonlocal n_hinted
        if outcome.extras.get("hint"):
            n_hinted += 1

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

    # The NO HINT rate is the arm's headline diagnostic: a run that hinted on almost nothing
    # scores like the baseline for reasons that have nothing to do with hint quality.
    n_rows = max(len(results) - n_completed, 1)
    log_info(f"Wrote a hint on {n_hinted}/{n_rows} episodes this run "
             f"({100.0 * n_hinted / n_rows:.0f}%).")
