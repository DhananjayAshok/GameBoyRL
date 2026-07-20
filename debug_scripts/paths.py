"""
Single source of truth for every path the debug commands read or write.

All derivations mirror the producing scripts exactly:
  - ``model_save_name`` follows propose_tasks_zeroshot.py (``model_name.split("/")[-1]``)
  - attempts/practice suffixes follow propose_and_attempt_all.sh and practice_tasks.py
  - the fine-tuned served-model name follows serve_and_benchmark.sh

Every accessor that reads a pipeline artifact goes through :meth:`Paths.require`, which
raises (via ``log_error``) naming both the missing path and the script that produces it.
Nothing here reads ``logs/``, wandb, or slurm — only artifacts the pipeline guarantees.
"""

import os

from utils import load_parameters, log_error


# Which script to point the user at when an artifact is missing.
PRODUCERS = {
    "grouped": "scripts/rl/create_all_traj.sh",
    "curiosity": "scripts/vlm/infer_tasks.sh (via scripts/pipeline/curiosity_all_tasks.sh)",
    "tasks": "scripts/vlm/propose_zeroshot.sh",
    "attempts": "scripts/vlm/attempt_tasks.sh",
    "guidance": "scripts/vlm/infer_guidance.sh",
    "practice": "scripts/vlm/practice_tasks.sh",
    "clean": "scripts/vlm/clean_practice.sh",
    "dataset": "scripts/vlm/create_dataset.sh",
    "benchmark": "scripts/benchmark.sh (via scripts/pipeline/serve_and_benchmark.sh)",
}

GROUPED_FILENAME = "grouped_global_high_reward_trajectories.pkl"

EXTRA_SUFFIXES = {
    "none": "",
    "zeroshot": "_prior_zeroshot",
    "curiosity": "_prior_curiosity",
    "zeroshot_with_curiosity": "_prior_zeroshot_with_curiosity",
}


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
                 model_name=None, output_dir=None):
        self.parameters = load_parameters(parameters)
        self.game = game
        self.run_name = run_name
        self.executor = executor
        self.model_name = model_name
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
        """Served-model name a fine-tuned checkpoint is benchmarked under (serve_and_benchmark.sh)."""
        return f"{self.model_save_name}-{self.game}-{self.run_name}"

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
    # Task proposal / attempt / practice
    # ------------------------------------------------------------------

    def proposed_tasks_dir(self) -> str:
        return os.path.join(self.storage_dir, "proposed_tasks", self.game, self.model_save_name)

    def curiosity_annotation(self) -> str:
        return os.path.join(self.proposed_tasks_dir(), "curiosity", self.run_name,
                            "trajectory_annotation.json")

    def zeroshot_dir(self) -> str:
        return os.path.join(self.proposed_tasks_dir(), "zeroshot")

    def tasks_file(self, extra: str = "zeroshot_with_curiosity") -> str:
        if extra not in EXTRA_SUFFIXES:
            log_error(f"Unknown --extra '{extra}'. Choose from {sorted(EXTRA_SUFFIXES)}.", self.parameters)
        return os.path.join(self.zeroshot_dir(), f"zeroshot_tasks{EXTRA_SUFFIXES[extra]}.jsonl")

    def available_extras(self) -> list[str]:
        """Every --extra variant whose tasks jsonl is actually on disk."""
        return [e for e in EXTRA_SUFFIXES if os.path.exists(self.tasks_file(e))]

    def attempts_dir(self, extra: str = "zeroshot_with_curiosity") -> str:
        stem = self.tasks_file(extra)[: -len(".jsonl")]
        return f"{stem}_{self.executor}_attempts"

    def all_trajectories_csv(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.attempts_dir(extra), "all_trajectories.csv")

    def success_trajectories_json(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.attempts_dir(extra), "success_trajectories.json")

    def success_trajectories_pkl(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.attempts_dir(extra), "success_trajectories.pkl")

    def guidance_json(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.attempts_dir(extra), "success_trajectories_guidance.json")

    def practice_dir(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.attempts_dir(extra), f"practice_{self.executor}")

    def practice_results_csv(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.practice_dir(extra), "results.csv")

    def clean_decisions_csv(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.practice_dir(extra), "clean_decisions.csv")

    def paraphrases_json(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.practice_dir(extra), "paraphrases.json")

    def practice_episode_pkl(self, group_idx, attempt, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.practice_dir(extra), f"{group_idx}_{attempt}.pkl")

    def train_csv(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.practice_dir(extra), "train_dataset.csv")

    def validation_csv(self, extra: str = "zeroshot_with_curiosity") -> str:
        return os.path.join(self.practice_dir(extra), "validation_dataset.csv")

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

    def video_path(self, bench_game: str, model: str, task: str) -> str | None:
        """
        Latest recorded video for one benchmarked task, or None if absent.

        Session layout (run_benchmark.py + GameBoyWorlds):
        ``<gbw>/sessions/<game>/benchmark_zero_shot_<executor>_<model>/<task_str>/<n>_<hash>/videos/0.mp4``

        Unlike the per-call PNGs, this path is keyed on the *model*, so base and fine-tuned
        runs do not overwrite each other. Repeated runs of the same model leave several
        ``<n>_<hash>`` dirs; the most recently modified one is returned.
        """
        gbw = self.gameboy_worlds_storage()
        if gbw is None:
            return None
        task_str = task.replace(" ", "_").lower()
        session = os.path.join(
            gbw, "sessions", bench_game,
            f"benchmark_zero_shot_{self.executor}_{model.lower()}", task_str,
        )
        if not os.path.isdir(session):
            return None
        runs = [os.path.join(session, d) for d in os.listdir(session)]
        runs = [d for d in runs if os.path.isdir(d)]
        if not runs:
            return None
        latest = max(runs, key=os.path.getmtime)
        video = os.path.join(latest, "videos", "0.mp4")
        return video if os.path.exists(video) else None
