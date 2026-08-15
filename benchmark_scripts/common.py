"""
Machinery shared by every benchmark arm.
"""

from __future__ import annotations

import gzip
import os
import pickle
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
from tqdm import tqdm

from gameboy_worlds import get_test_environment

from execution.info_doc import load_document
from python_scripts import paths
from utils import depathify, log_error, log_info


# Columns every arm writes, in this order. Arm-specific columns are appended after these,
# and SESSION_COLUMN last, so a positional reader of the common prefix works on any arm's
COMMON_COLUMNS = [
    "game",
    "task",
    "success",
    "n_steps",
    "n_invalid",
    # Supervisor and executor cost are kept apart rather than summed: the comparison the
    # arms exist to make is what the supervision itself costs, and an arm that plans well
    # but spends more on planning than it saves on execution is not a win. A single total
    # cannot show that, and a reader who wants one can add two columns. Empty where the
    # backend did not report usage, which is a different fact from zero.
    "supervisor_input_tokens",
    "supervisor_output_tokens",
    "executor_input_tokens",
    "executor_output_tokens",
    "subgoals_reached",
    "all_subgoals",
    "report",
]

# Points at the emulator session directory for this episode, which is where both the
# recorded video and the archived executor report live. Always the last column.
SESSION_COLUMN = "session_dirs"

#: Pickled per-episode executor report(s), written into the session dir beside videos/0.mp4.
REPORT_FILENAME = "report.pkl.gz"


# ---------------------------------------------------------------------------
# Task selection, paths, resume
# ---------------------------------------------------------------------------


def select_tasks(tasks: pd.DataFrame, n_tasks: Optional[int]) -> pd.DataFrame:
    """The first ``n_tasks`` benchmark tasks, or all of them when ``n_tasks`` is None.

    A prefix rather than a sample, so a quick look extends into a bigger run: the tasks in
    ``--n_tasks 5`` are the first five of ``--n_tasks 10``, and both are a prefix of the
    full sweep. That nesting is what makes partial runs comparable to each other and to the
    whole; a random subsample of 5 and of 10 share no such relationship.
    """
    if n_tasks is None:
        return tasks
    if not (1 <= n_tasks <= len(tasks)):
        raise ValueError(f"n_tasks must be between 1 and {len(tasks)}, got {n_tasks}")
    return tasks.head(n_tasks).reset_index(drop=True)


def results_path(parameters: dict, game: str, *, supervisor: str, executor: str,
                 controller_variant: str, model: str, n_tasks: Optional[int]) -> str:
    """Where this run's CSV goes, creating the directory.

    A subset run gets its own file: resuming a full sweep from a 5-task CSV would read the
    first five as done and silently skip them.

    Delegates to :func:`python_scripts.paths.benchmark_csv` rather than building the name
    here. This used to be an f-string, and ``Paths.benchmark_csv`` rebuilt it without the
    supervisor prefix or the ``_firstN`` suffix — so ``debug.py benchmark`` could not open
    any CSV this function had ever written. The supervisor and n_tasks are passed through
    instead of being pre-baked into a ``stem`` string so only one place knows the order.
    """
    return paths.benchmark_csv(parameters, game=game, supervisor=supervisor,
                               executor=executor, controller_variant=controller_variant,
                               model=model, n_tasks=n_tasks, create=True)


def load_checkpoint(save_path: str, regenerate: bool, columns: list,
                    parameters: dict) -> tuple:
    """Rows already written for this run, and how many tasks they cover.

    Refuses to resume a CSV whose column count differs from what this arm writes. Feeding
    mismatched rows to ``pd.DataFrame(..., columns=columns)`` raises anyway, but with a
    pandas shape error that says nothing about the cause — which is nearly always a file
    written before a column was added.
    """
    if regenerate or not os.path.exists(save_path):
        return [], 0

    existing = pd.read_csv(save_path)
    if len(existing.columns) != len(columns):
        log_error(
            f"Cannot resume {save_path}: it has {len(existing.columns)} columns but this "
            f"run writes {len(columns)}. It was written by an older version. Pass "
            f"--regenerate to overwrite it, or move it aside to keep it.",
            parameters,
        )
    log_info(f"Resuming from checkpoint: {len(existing)} tasks already completed "
             f"in {save_path}")
    return existing.values.tolist(), len(existing)


# ---------------------------------------------------------------------------
# Knowledge loading (info and plan arms)
# ---------------------------------------------------------------------------


def load_documents(info_docs: str, parameters: dict) -> list:
    """Load each --info_docs path. Each document carries its own provenance label."""
    documents = []
    for path in [p.strip() for p in info_docs.split(",") if p.strip()]:
        if not os.path.exists(path):
            log_error(f"info document not found at {path}. Produced by: "
                      "scripts/vlm/build_info.sh", parameters)
        document = load_document(path, parameters=parameters)
        label = document.provenance.label or "(no provenance recorded)"
        documents.append(document)
        log_info(f"Loaded info document '{label}' — {len(document.task_entries)} task / "
                 f"{len(document.image_entries)} image entries ({path})")
    if not documents:
        log_error("No usable --info_docs given.", parameters)
    return documents


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


@dataclass
class PlayResult:
    """What an arm hands back after driving one attempt at a task.

    :param report: This episode's :class:`~execution.report.SupervisorReport` — every
        supervisor call and every executor leg, interleaved. The archived artifact, and the
        only place the frames exist.
    :param extras: Arm-specific values, keyed by name, for that arm's extra columns.

    ``report_str``, ``n_invalid`` and ``legs`` used to be separate fields here. All three are
    now properties of the report (``str(report)``, ``report.n_invalid``,
    ``report.executor_reports``), and carrying them alongside meant the same fact could be
    recorded twice and disagree.
    """

    report: Any
    extras: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Whether the environment signalled the task complete on the last executor leg.

        The environment's own verdict, never a supervisor's judgement — it is the pipeline's
        only ground-truth success signal. An episode with no executor leg (a supervisor that
        produced nothing) is a failure.
        """
        legs = self.report.executor_reports if self.report is not None else []
        return bool(legs) and legs[-1].termination_reason == "terminated"


@dataclass
class EpisodeOutcome:
    """The per-task result every arm produces, whatever drove the episode."""

    success: bool = False
    n_steps: int = 0
    n_invalid: int = 0
    #: Token cost of the episode, read off the report in :func:`run_episode` exactly as
    #: ``n_invalid`` is. None means the backend did not report usage — kept distinct from
    #: 0, which is what a supervisor that genuinely made no calls (the baseline arm's
    #: DummySupervisor) reports.
    supervisor_input_tokens: Optional[int] = None
    supervisor_output_tokens: Optional[int] = None
    executor_input_tokens: Optional[int] = None
    executor_output_tokens: Optional[int] = None
    subgoals_reached: list = field(default_factory=list)
    subgoals_all: Any = None
    error: bool = True
    report_str: Optional[str] = None
    session_dirs: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def session_path_of(environment) -> Optional[str]:
    """The emulator instance directory this environment records into.

    ``<storage>/sessions/<game>/<session_name>/<task>/<n>_<hash>/``, holding ``videos/0.mp4``
    and — once :func:`save_report` has run — the archived report. Read from the emulator
    rather than reconstructed from the naming convention, because the trailing ``<n>_<hash>``
    is chosen at construction time and a convention-based guess has to fall back to "the most
    recently modified directory", which is wrong as soon as a task has been run twice.

    Reaches through ``Environment._emulator``. Returns None rather than raising if that
    attribute ever moves, since losing the archive should not fail a benchmark run.
    """
    emulator = getattr(environment, "_emulator", None)
    return getattr(emulator, "session_path", None) if emulator is not None else None


def save_report(session_path: str, *, supervisor: str, row, executor_name: str, model: str,
                report) -> Optional[str]:
    """Archive this episode's :class:`~execution.report.SupervisorReport` beside its video.

    The report is what makes an episode reconstructable after the fact. Its ``event_log``
    interleaves the supervisor's own calls with the full report of every executor leg, and
    each call record — either kind — carries the images it saw, its prompt and the raw
    response. Nothing else on disk has the frames: the CSV's ``report`` column is a rendered
    string, and the executor's own PNGs are written only under ``--verbose`` into a directory
    keyed on the executor class, which one run overwrites for the next.

    Written gzipped, since always-on archiving across a full sweep is otherwise gigabytes.
    Compression is worth far more on the prompt text than on the frames, so the ratio
    depends on how image-heavy the run is.

    Self-describing rather than a bare report, so a reader does not need to know which arm or
    which run wrote it — the benchmark CSV and this file are otherwise linked only by a
    directory path.

    Failure here is logged and swallowed: an unwritable archive must not cost an episode
    that has already been paid for in VLM calls.
    """
    payload = {
        "supervisor": supervisor,
        "game": row["game"],
        "task": row["task"],
        "init_state": row.get("init_state"),
        "executor": executor_name,
        "model": model,
        "report": report,
    }
    path = os.path.join(session_path, REPORT_FILENAME)
    try:
        with gzip.open(path, "wb", compresslevel=6) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path
    except Exception as error:  # noqa: BLE001 - archiving must never fail a run
        print(f"WARNING: could not write report to {path}: {error}")
        return None


def run_episode(row, play: Callable[[Any], PlayResult], *, supervisor: str,
                controller_variant: str, executor_name: str, model: str,
                **emulator_kwargs) -> EpisodeOutcome:
    """Run one benchmark task, once.

    Owns everything that is the same for every arm: subgoal accumulation, step totals,
    archiving, environment teardown and error trapping. ``play(environment)`` is the only
    part an arm supplies — it drives the episode and reports what happened, including its
    own verbose output, since what is worth printing differs by arm.

    One attempt per task, deliberately. A whole-task retry built a second emulator session
    — hence a second video and a second report — under one CSV row, so the row described a
    single attempt while the session directory held several and nothing said which was
    which. Arms that retry *internally* (the plan arm's --max_attempts_per_step and
    --max_replans) are unaffected: those stay inside one session.

    An exception anywhere leaves ``error=True`` and whatever was accumulated up to that
    point: the sweep stops on first error. 
    """
    outcome = EpisodeOutcome()
    mission = row["task"]
    # depathify, not a bare space-replace: a task carrying a "/" ("Reach the BUY/SELL
    # choice menu…", "…first stats page (HP/MP)") would otherwise be spliced into the
    # session path as a directory separator, burying that episode one level deeper than
    # every other task's.
    task_str = depathify(mission).lower()
    emulator_kwargs = dict(emulator_kwargs)
    emulator_kwargs["session_name"] += f"/{task_str}/"
    emulator_kwargs["wait_ticks"] = 20

    try:
        environment = get_test_environment(
            row=row, controller_variant=controller_variant, **emulator_kwargs
        )
        # Captured before close(): the emulator removes its session directory on close
        # when nothing was written into it.
        session_path = session_path_of(environment)

        result = play(environment)

        last_state = environment.get_info()
        outcome.subgoals_all = last_state["subgoals"]["all"]
        for subgoal in last_state["subgoals"]["completed"]:
            if subgoal not in outcome.subgoals_reached:
                outcome.subgoals_reached.append(subgoal)

        outcome.n_steps += last_state["core"]["steps"]
        outcome.n_invalid += result.report.n_invalid if result.report is not None else 0
        outcome.report_str = str(result.report) if result.report is not None else None
        if result.report is not None:
            outcome.supervisor_input_tokens = result.report.supervisor_input_tokens
            outcome.supervisor_output_tokens = result.report.supervisor_output_tokens
            outcome.executor_input_tokens = result.report.executor_input_tokens
            outcome.executor_output_tokens = result.report.executor_output_tokens
        outcome.extras = result.extras

        if session_path is not None:
            save_report(session_path, supervisor=supervisor, row=row,
                        executor_name=executor_name,
                        model=model, report=result.report)
            outcome.session_dirs.append(session_path)

        environment.close()
        outcome.error = False
        outcome.success = result.success

    except Exception as error:  # noqa: BLE001 - one bad task must not end the sweep
        outcome.error = True
        print(f"Error during execution of task '{mission}': {error}")
        traceback.print_exc()

    return outcome


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def common_row(row, outcome: EpisodeOutcome) -> list:
    """The :data:`COMMON_COLUMNS` values for one finished episode."""
    return [
        row["game"],
        row["task"],
        outcome.success,
        outcome.n_steps,
        outcome.n_invalid,
        outcome.supervisor_input_tokens,
        outcome.supervisor_output_tokens,
        outcome.executor_input_tokens,
        outcome.executor_output_tokens,
        outcome.subgoals_reached,
        outcome.subgoals_all,
        outcome.report_str,
    ]


def run_sweep(tasks: pd.DataFrame, *, columns: list, save_path: str, results: list,
              n_completed: int,
              run_one: Callable[[Any], EpisodeOutcome],
              build_row: Callable[[Any, EpisodeOutcome], list],
              on_episode: Optional[Callable[[Any, EpisodeOutcome], None]] = None) -> list:
    """Run every task and write the CSV after each one.

    Flushing per episode rather than at the end is deliberate: these runs are long enough
    that they get pre-empted, and a partial CSV is what resume reads.
    """
    for i, row in tqdm(tasks.iterrows(), total=len(tasks)):
        if i < n_completed:
            continue

        outcome = run_one(row)
        if outcome.error:
            log_error(f"Error occurred during execution of task '{row['task']}' "
                      "- exiting loop")
        if on_episode is not None:
            on_episode(row, outcome)

        results.append(build_row(row, outcome))
        pd.DataFrame(results, columns=columns).to_csv(save_path, index=False)
        log_info(f"Saved benchmark results to {save_path}")
    return results
