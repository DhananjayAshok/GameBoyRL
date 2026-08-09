"""
Called by scripts/benchmark_info.sh. Use --help for CLI options.

The context-engineering arm's benchmark runner: a sibling of run_benchmark.py that constructs
an InfoHintSupervisor per task instead of an executor directly. The supervisor reads a
prebuilt info document (or, in --mode init_state, the stage-A insights), writes one hint for
the opening screen, and runs the executor with it.

run_benchmark.py is deliberately left untouched. Rows written here have the same columns, so
viz.py reads both arms' CSVs identically — with two extra columns (hint, n_entries_selected)
appended, which pandas-based readers ignore and which make a results file self-explaining.
"""

import json
import os
import traceback

import click
import pandas as pd
from tqdm import tqdm

from gameboy_worlds import AVAILABLE_GAMES, get_benchmark_tasks, get_test_environment

from execution.info_doc import Provenance, load_document
from execution.registry import AVAILABLE_EXECUTORS
from execution.supervisors import InfoHintSupervisor
from utils import load_parameters, log_error, log_info


def _load_documents(info_docs, parameters):
    """Load each --info_docs path. Each document carries its own provenance label."""
    documents = []
    for path in [p.strip() for p in info_docs.split(",") if p.strip()]:
        if not os.path.exists(path):
            log_error(f"info document not found at {path}. Produced by: scripts/vlm/build_info.sh",
                      parameters)
        document = load_document(path, parameters=parameters)
        label = document.provenance.label or "(no provenance recorded)"
        documents.append(document)
        log_info(f"Loaded info document '{label}' — {len(document.task_entries)} task / "
                 f"{len(document.image_entries)} image entries ({path})")
    if not documents:
        log_error("No usable --info_docs given.", parameters)
    return documents


def _load_insight_rows(insights_paths, parameters):
    """
    Read and union the stage-A insights.jsonl files used by --mode init_state.

    Every row is tagged with the provenance label its own leaf document records, the same
    labelling the retrieval mode applies to document entries. This matters as soon as more
    than one source is unioned: the hint writer is told which insights came from verified
    solutions (zeroshot/attempt) and which from exploration labels (curiosity), and
    ``group_idx`` alone cannot express that — the two verticals number their groups
    independently, so their keys overlap and mean different things.
    """
    rows = []
    for path in [p.strip() for p in insights_paths.split(",") if p.strip()]:
        if not os.path.exists(path):
            log_error(f"insights.jsonl not found at {path}. Produced by: "
                      "scripts/vlm/build_info.sh --stage a", parameters)
        label = ""
        n_before = len(rows)
        with open(path, "r") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    label = Provenance.from_dict(
                        (row.get("document") or {}).get("provenance")).label
                    row["source"] = label
                    rows.append(row)
        states = sorted({r.get("init_state") for r in rows[n_before:] if r.get("init_state")})
        log_info(f"Loaded {len(rows) - n_before} insight rows from '{label}' "
                 f"covering {len(states)} init_states ({path})")
    if not rows:
        log_error("No usable --insights_paths given.", parameters)
    all_states = sorted({r.get("init_state") for r in rows if r.get("init_state")})
    log_info(f"Total {len(rows)} stage-A insight rows covering init_states: {all_states}")
    return rows


def run_task(row, max_resets, controller_variant, executor_class, max_tool_calls,
             documents, insight_rows, mode, hint_vlm_model, hint_vlm_kind,
             max_concurrency, vlm_model=None, vlm_kind=None, verbose=False, **emulator_kwargs):
    """One benchmark task: build the hint from the opening screen, then run the executor."""
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
    hint = None
    n_selected = 0
    selected_ids = []
    try:
        while n_resets < max_resets + 1:
            environment = get_test_environment(
                row=row, controller_variant=controller_variant, **emulator_kwargs
            )
            supervisor = InfoHintSupervisor(
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
                hint_vlm_model=hint_vlm_model,
                hint_vlm_kind=hint_vlm_kind,
                max_concurrency=max_concurrency,
                vlm_model=vlm_model,
                vlm_kind=vlm_kind,
            )
            report = supervisor.evaluate()
            hint = supervisor.hint
            # Which entries the hint was synthesised from, kept per episode so a hint can be
            # traced back to its evidence from the CSV alone. In init_state mode each id
            # pins the exact stage-A row (source/group_idx#category) in insights.jsonl.
            selected_ids = list(supervisor.selected_ids)
            n_selected = len(selected_ids)
            if verbose:
                print(f"\n  Hint for '{mission}': {hint}")
                print(f"  Synthesised from {n_selected} entr(ies): {selected_ids}")

            last_state = environment.get_info()
            if subgoals_all is None:
                subgoals_all = last_state["subgoals"]["all"]
            for subgoal in last_state["subgoals"]["completed"]:
                if subgoal not in subgoals_reached:
                    subgoals_reached.append(subgoal)

            n_steps_total += last_state["core"]["steps"]
            n_invalid_total += len(report.invalid_steps)

            report_str = str(report)
            if verbose:
                print(f"\n  Reset {n_resets} trajectory:")
                report.show()

            environment.close()
            error = False
            if report.termination_reason == "terminated":
                success = True
                break
            n_resets += 1

    except Exception as e:
        error = True
        print(f"Error during execution of task '{mission}': {e}")
        traceback.print_exc()
    return (success, n_resets - 1, n_steps_total, n_invalid_total, subgoals_reached,
            subgoals_all, error, report_str, hint, n_selected, selected_ids)


@click.command()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--info_docs", default=None, type=str,
              help="Comma-separated info.json path(s). Entries from all of them are unioned and "
                   "tagged with a source label. Required for --mode retrieval.")
@click.option("--insights_paths", default=None, type=str,
              help="Comma-separated insights.jsonl path(s). Required for --mode init_state.")
@click.option("--mode", default="retrieval", type=click.Choice(["retrieval", "init_state"]),
              help="How the hint's candidate knowledge is selected. 'init_state' skips the "
                   "document and keys off the episode's init state (the retrieval-free upper bound).")
@click.option("--hint_vlm_model", default=None, type=str,
              help="Model for the relevance and hint-writing calls. Defaults to the executor's.")
@click.option("--hint_vlm_kind", default=None, type=str)
@click.option("--max_concurrency", default=8, show_default=True, type=int,
              help="Parallel relevance calls per episode; they are independent.")
@click.option("--controller_variant", default="low_level", type=str)
@click.option("--executor", default="simple", type=click.Choice(list(AVAILABLE_EXECUTORS.keys())))
@click.option("--executor_vlm_model", default=None, type=str)
@click.option("--executor_vlm_kind", default=None, type=str)
@click.option("--save_video", type=bool, default=True)
@click.option("--max_resets", default=1, type=int)
@click.option("--max_steps", default=200, type=int)
@click.option("--max_tool_calls", default=0, type=int)
@click.option("--override_index", default=None, type=int, required=False)
@click.option("--random_sample", type=int, default=None)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--regenerate", is_flag=True, default=False)
def do(game, info_docs, insights_paths, mode, hint_vlm_model, hint_vlm_kind, max_concurrency,
       controller_variant, executor, executor_vlm_model, executor_vlm_kind, save_video,
       max_resets, max_steps, max_tool_calls, override_index, random_sample, verbose, regenerate):
    project_parameters = load_parameters()
    vlm_name = executor_vlm_model or project_parameters["executor_vlm_model"]
    model_save_name = vlm_name.split("/")[-1].lower()

    documents, insight_rows = None, None
    if mode == "retrieval":
        if not info_docs:
            log_error("--mode retrieval requires --info_docs.", project_parameters)
        documents = _load_documents(info_docs, project_parameters)
    else:
        if not insights_paths:
            log_error("--mode init_state requires --insights_paths.", project_parameters)
        insight_rows = _load_insight_rows(insights_paths, project_parameters)

    session_name = f"benchmark_info_{mode}_{executor}_{model_save_name}"
    emulator_kwargs = {
        "headless": True,
        "save_video": save_video,
        "session_name": session_name,
        "max_steps": max_steps,
    }
    benchmark_tasks = get_benchmark_tasks(game=game)
    columns = [
        "game",
        "task",
        "success",
        "n_resets",
        "n_steps",
        "n_invalid",
        "subgoals_reached",
        "all_subgoals",
        "report",
        "hint",
        "n_entries_selected",
        "selected_entry_ids",
    ]
    if random_sample is not None:
        if not (1 <= random_sample <= len(benchmark_tasks) - 1):
            raise ValueError(
                f"random_sample must be between 1 and {len(benchmark_tasks) - 1}, got {random_sample}"
            )
        benchmark_tasks = benchmark_tasks.sample(n=random_sample, random_state=42).reset_index(drop=True)

    results_dir = project_parameters["results_dir"]
    os.makedirs(f"{results_dir}/benchmark/{game}/", exist_ok=True)
    stem = f"info_{mode}_{executor}_{model_save_name}"
    if random_sample is not None:
        stem += f"_sample{random_sample}"
    save_path = f"{results_dir}/benchmark/{game}/{stem}.csv"

    if not regenerate and os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        results = existing_df.values.tolist()
        n_completed = len(existing_df)
        print(f"Resuming from checkpoint: {n_completed} tasks already completed in {save_path}")
    else:
        results = []
        n_completed = 0

    n_hinted = 0
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
         report_str, hint, n_selected, selected_ids) = run_task(
            row=row,
            max_resets=max_resets,
            controller_variant=controller_variant,
            executor_class=AVAILABLE_EXECUTORS[executor],
            max_tool_calls=max_tool_calls,
            documents=documents,
            insight_rows=insight_rows,
            mode=mode,
            hint_vlm_model=hint_vlm_model or executor_vlm_model,
            hint_vlm_kind=hint_vlm_kind or executor_vlm_kind,
            max_concurrency=max_concurrency,
            vlm_model=executor_vlm_model,
            vlm_kind=executor_vlm_kind,
            verbose=verbose,
            **emulator_kwargs,
        )
        if error:
            log_error(f"Error occurred during execution of task '{row['task']}' - exiting loop")
        if hint:
            n_hinted += 1

        results.append([
            row["game"], row["task"], success, n_resets, n_steps, n_invalid,
            subgoals_reached, subgoals_all, report_str, hint, n_selected,
            # JSON rather than a Python list repr so it survives the CSV round-trip as
            # something machine-readable: json.loads(row.selected_entry_ids) just works.
            json.dumps(selected_ids),
        ])
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(save_path, index=False)
        log_info(f"Saved benchmark results to {save_path}")

    # The NO HINT rate is the arm's headline diagnostic: a run that hinted on almost nothing
    # scores like the baseline for reasons that have nothing to do with hint quality.
    n_rows = max(len(results) - n_completed, 1)
    log_info(f"Wrote a hint on {n_hinted}/{n_rows} episodes this run "
             f"({100.0 * n_hinted / n_rows:.0f}%).")


if __name__ == "__main__":
    do()
