"""
The pipeline's path scheme: one implementation, used by producers and consumers alike.

Every artifact path is a pure function of a small identity — ``(storage_dir, game,
model_name, run_name, executor)`` plus a source vertical. This module owns that function,
in both directions: :class:`Paths` builds a path from an identity, and
:func:`source_label` recovers the vertical from a path that was built.

It used to live in ``debug_scripts/``, described as mirroring the producers — its own
docstring said *"All derivations mirror the producing scripts exactly"*. Mirroring by hand
is the failure mode: the producers built their paths with f-strings, the debug commands
rebuilt them here, and Bash built them a third way. A change to one silently
desynchronised the rest, and the symptom was a debug command reporting a missing artifact
for a run that finished fine. The producers now call this module, so there is nothing left
to mirror.

``tests/test_paths.py`` pins the scheme: a frozen fixture of every accessor over the full
input grid, a differential against the Bash helpers in ``scripts/core/utils.sh``, and a
check against artifacts actually on disk. Run it after touching anything here.

Every accessor that reads a pipeline artifact goes through :meth:`Paths.require`, which
raises (via ``log_error``) naming both the missing path and the script that produces it.
Nothing here reads ``logs/``, wandb, or slurm — only artifacts the pipeline guarantees.
"""

import os

# Submodule imports rather than ``from utils import ...``: this module is imported by
# producers that have no reason to pull in the VLM stack that utils/__init__ carries.
from utils.log_handling import log_error
from utils.parameter_handling import load_parameters


# Which script to point the user at when an artifact is missing.
PRODUCERS = {
    "grouped": "scripts/rl/create_all_traj.sh",
    "curiosity": "scripts/vlm/infer_tasks.sh (via scripts/pipeline/curiosity_all_tasks.sh)",
    "tasks": "scripts/vlm/propose_zeroshot.sh",
    "attempts": "scripts/vlm/attempt_tasks.sh",
    "benchmark": "scripts/benchmark.sh (via scripts/pipeline/serve_and_benchmark.sh)",
    "info": "scripts/vlm/build_info.sh",
    "insights": "scripts/vlm/build_info.sh (stage A)",
}

GROUPED_FILENAME = "grouped_global_high_reward_trajectories.pkl"
INFO_DOC_FILENAME = "info.json"
INSIGHTS_FILENAME = "insights.jsonl"


def source_label(artifact_dir: str) -> str:
    """
    Short provenance label for an artifact dir, recovered from its path.

    The **inverse** of the scheme :class:`Paths` builds, which is why it lives beside it.
    The two verticals lay their directories out differently::

        curiosity  .../curiosity/<run_name>/info_<model>_<executor>    -> "curiosity"
        zeroshot   .../zeroshot/zeroshot_tasks_<executor>_attempts/... -> "zeroshot"

    Prefer a *recorded* label where one exists — an info document carries its own
    provenance (see :class:`~execution.info_doc.Provenance`) and should be read, not
    re-derived. This function is the fallback for artifacts that record nothing.

    Falls back to the parent directory name for anything unrecognised, so an unusual
    layout still produces a distinguishable label rather than crashing.
    """
    parts = os.path.normpath(artifact_dir).split(os.sep)
    if len(parts) >= 3 and parts[-3] == 'curiosity':
        return 'curiosity'
    stem = parts[-2] if len(parts) >= 2 else os.path.basename(artifact_dir)
    if stem.endswith('_attempts'):
        stem = stem[: -len('_attempts')]
        stem = stem.rsplit('_', 1)[0]  # drop the trailing _<executor>
    if stem.startswith('zeroshot_tasks'):
        return 'zeroshot'
    return stem or 'unknown'


class Paths:
    """
    Resolves every debug-relevant path for one (game, run_name, executor, model) identity.

    :param parameters: Loaded project parameters. If None, loaded from config.
    :param game: Game name, e.g. ``"deja_vu_1"``.
    :param run_name: Run name used by the RL/curiosity stages, e.g. ``"my_run"``.
    :param executor: Executor short name, e.g. ``"history"``.
    :param model_name: Full VLM name, e.g. ``"google/gemma-4-31b-it"``. Only needed for
        stages whose paths are keyed on the model; ``None`` is fine for curiosity.
    :param output_dir: Root for generated reports. Defaults to ``<results_dir>/debug/<game>``.
    """

    def __init__(self, parameters=None, game=None, run_name="my_run", executor="history",
                 model_name=None, output_dir=None, mode="both"):
        self.parameters = load_parameters(parameters)
        self.game = game
        self.run_name = run_name
        self.executor = executor
        self.model_name = model_name
        self.mode = mode
        self.storage_dir = self.parameters["storage_dir"]
        self.results_dir = self.parameters["results_dir"]
        self._output_dir = output_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def model_save_name(self) -> str:
        """The basename the pipeline uses in paths (``google/gemma-4-31b-it`` -> ``gemma-4-31b-it``)."""
        if self.model_name is None:
            log_error("model_name is required for this command but was not provided.", self.parameters)
        return self.model_name.split("/")[-1]

    @property
    def finetuned_model_name(self) -> str:
        """
        Served-model name a fine-tuned checkpoint is benchmarked under.

        Mirrors serve_and_benchmark.sh's ``${model_save_name}-${game}-${run_name}-${mode}``.
        """
        return f"{self.model_save_name}-{self.game}-{self.run_name}-{self.mode}"

    def require(self, path: str, kind: str) -> str:
        """Return *path* if it exists, otherwise raise naming the producing script."""
        if not os.path.exists(path):
            log_error(
                f"Required artifact not found: {path}\n"
                f"  This is produced by: {PRODUCERS.get(kind, kind)}\n"
                f"  Run that stage for game '{self.game}' before this debug command.",
                self.parameters,
            )
        return path

    def debug_dir(self, stage: str, *sub: str) -> str:
        """``<results_dir>/debug/<game>/<stage>[/sub...]``, created on demand."""
        root = self._output_dir or os.path.join(self.results_dir, "debug", self.game)
        path = os.path.join(root, stage, *sub)
        os.makedirs(path, exist_ok=True)
        return path

    def debug_frames_dir(self, stage: str, *sub: str) -> str:
        """``<storage_dir>/tmp/debug_frames/<game>/<stage>[/sub...]``, created on demand.

        On storage rather than beside the markdown in :meth:`debug_dir`, because a
        frame-by-frame report is thousands of PNGs and ``results_dir`` is on the home
        filesystem, where the inode quota bites long before the disk quota does.

        Keyed on game and stage, with callers appending model and task below that, so two
        runs cannot overwrite each other's frames — which is exactly the failure the
        executor's own ``--verbose`` PNGs have, being keyed on the executor class.
        """
        path = os.path.join(self.storage_dir, "tmp", "debug_frames", self.game, stage, *sub)
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # Curiosity / grouping
    # ------------------------------------------------------------------

    def grouped_dir(self) -> str:
        return os.path.join(self.storage_dir, "grouped_trajectories", self.game, self.run_name)

    def grouped_file(self, init_state: str) -> str:
        return os.path.join(self.grouped_dir(), init_state, GROUPED_FILENAME)

    def init_states(self) -> list[str]:
        """Every init_state with a grouped-trajectory file, sorted. Raises if none exist."""
        root = self.require(self.grouped_dir(), "grouped")
        states = sorted(
            d for d in os.listdir(root)
            if os.path.exists(self.grouped_file(d))
        )
        if not states:
            log_error(
                f"No grouped trajectory files under {root}.\n"
                f"  Produced by: {PRODUCERS['grouped']}",
                self.parameters,
            )
        return states

    # ------------------------------------------------------------------
    # Task proposal / attempt
    # ------------------------------------------------------------------

    def proposed_tasks_dir(self) -> str:
        return os.path.join(self.storage_dir, "proposed_tasks", self.game, self.model_save_name)

    def curiosity_annotation(self) -> str:
        return os.path.join(self.proposed_tasks_dir(), "curiosity", self.run_name,
                            "trajectory_annotation.json")

    def zeroshot_dir(self) -> str:
        return os.path.join(self.proposed_tasks_dir(), "zeroshot")

    def tasks_file(self) -> str:
        return os.path.join(self.zeroshot_dir(), "zeroshot_tasks.jsonl")

    def attempts_dir(self) -> str:
        stem = self.tasks_file()[: -len(".jsonl")]
        return f"{stem}_{self.executor}_attempts"

    def all_trajectories_csv(self) -> str:
        return os.path.join(self.attempts_dir(), "all_trajectories.csv")

    def success_trajectories_json(self) -> str:
        return os.path.join(self.attempts_dir(), "success_trajectories.json")

    def success_trajectories_pkl(self) -> str:
        return os.path.join(self.attempts_dir(), "success_trajectories.pkl")

    # ------------------------------------------------------------------
    # Info documents (context-engineering vertical)
    # ------------------------------------------------------------------
    # build_info.py writes next to its own input stem, so these two accessors just
    # re-derive that rule per vertical.

    def info_dir(self) -> str:
        """The zeroshot/attempt vertical's info dir — attempts_dir/info_<model>_<executor>."""
        return os.path.join(self.attempts_dir(),
                            f"info_{self.model_save_name}_{self.executor}")

    def curiosity_info_dir(self) -> str:
        """Curiosity vertical's info dir — dirname(trajectory_annotation)/info_<model>_<executor>."""
        return os.path.join(os.path.dirname(self.curiosity_annotation()),
                            f"info_{self.model_save_name}_{self.executor}")

    def source_info_dir(self, source: str) -> str:
        """Info dir for a named source. ``source`` is 'attempt' or 'curiosity'."""
        if source == "curiosity":
            return self.curiosity_info_dir()
        if source == "attempt":
            return self.info_dir()
        log_error(f"Unknown --source '{source}'. Choose from ['attempt', 'curiosity'].",
                  self.parameters)

    def info_doc(self, source: str = "attempt") -> str:
        return os.path.join(self.source_info_dir(source), INFO_DOC_FILENAME)

    def parametric_doc(self) -> str:
        """The parametric document for this game and model.

        Keyed on game + model only, and deliberately not on run_name or executor: nothing
        about this document depends on a trajectory run or on which executor plays, because
        it is written from the model's priors before any of that exists. Putting it under
        the trajectory tree would imply a dependency it does not have, and would make the
        same document be regenerated once per run name.
        """
        return os.path.join(self.storage_dir, "parametric_docs", self.game,
                            self.model_save_name, INFO_DOC_FILENAME)

    def insights_jsonl(self, source: str = "attempt") -> str:
        return os.path.join(self.source_info_dir(source), INSIGHTS_FILENAME)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark_csv(self, bench_game: str, model: str) -> str:
        return os.path.join(self.results_dir, "benchmark", bench_game, f"{self.executor}_{model}.csv")

    def benchmark_series_csv(self) -> str:
        """The GameBoyWorlds series CSV whose `game` column contains self.game."""
        import pandas as pd
        tests_dir = os.path.join(self.parameters["project_root"], "GameBoyWorlds", "benchmark", "tests")
        self.require(tests_dir, "benchmark")
        for filename in sorted(os.listdir(tests_dir)):
            if not filename.endswith(".csv"):
                continue
            path = os.path.join(tests_dir, filename)
            if self.game in set(pd.read_csv(path)["game"].unique()):
                return path
        log_error(
            f"No series CSV under {tests_dir} contains game '{self.game}'.", self.parameters
        )

    def gameboy_worlds_storage(self) -> str | None:
        """GameBoyWorlds storage_dir, parsed from its config.env. None if unreadable."""
        env_path = os.path.join(self.parameters["project_root"], "GameBoyWorlds", "configs", "config.env")
        if not os.path.exists(env_path):
            return None
        with open(env_path) as handle:
            for line in handle:
                if line.strip().startswith("export storage_dir="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        return None

    # There is deliberately no video_path() accessor. A video is found through the episode
    # row's own ``session_dirs``, beside that episode's report.pkl.gz — see
    # ``debug_scripts/benchmark.py:_video_path``. Rebuilding the path from the task name
    # instead cannot address an episode: several benchmark rows share a task string, so they
    # share the task-named sessions directory too. The rebuilt name was also wrong for every
    # arm but the baseline, since the sessions directory is named for the arm that ran it
    # (``benchmark_info_subgoal_retrieval_<executor>_<model>``, not ``benchmark_dummy_…``).
