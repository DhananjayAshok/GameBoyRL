"""
Scaffolding check: resolve every artifact path for the given identity and report which
exist. Run this first when pointing the debug tools at a new game/run — it tells you
which stages have artifacts before any report tries to read them.
"""

import os

import click

from python_scripts.paths import BENCHMARK_SUPERVISORS, Paths
from utils import log_info


def _mark(path: str) -> str:
    return "OK  " if os.path.exists(path) else "MISS"


@click.command(name="check_paths")
@click.option("--model_name", default=None,
              help="Full VLM name (e.g. google/gemma-4-31b-it). Omit to check curiosity paths only.")
@click.pass_obj
def check_paths(obj, model_name):
    """Print every derived path with an exists/missing marker."""
    paths = Paths(
        parameters=obj["parameters"],
        game=obj["game"],
        run_name=obj["run_name"],
        executor=obj["executor"],
        controller_variant=obj["controller_variant"],
        extra_name=obj["extra_name"],
        model_name=model_name,
        output_dir=obj["output_dir"], mode=obj["mode"],
    )

    lines = [
        f"game       : {paths.game}",
        f"run_name   : {paths.run_name}",
        f"executor   : {paths.executor}",
        f"model_name : {model_name or '(not given)'}",
        f"storage_dir: {paths.storage_dir}",
        f"results_dir: {paths.results_dir}",
        "",
        "-- curiosity --",
    ]
    grouped = paths.grouped_dir()
    lines.append(f"  [{_mark(grouped)}] {grouped}")
    if os.path.exists(grouped):
        states = sorted(d for d in os.listdir(grouped) if os.path.exists(paths.grouped_file(d)))
        lines.append(f"         {len(states)} init_state(s) with grouped trajectories")

    if model_name is None:
        lines.append("\n(model_name not given — skipping model-keyed stages)")
        log_info("\n".join(lines), obj["parameters"])
        return

    lines += [
        "",
        "-- proposal / attempt --",
        f"  [{_mark(paths.curiosity_annotation())}] {paths.curiosity_annotation()}",
    ]
    tasks = paths.tasks_file()
    lines.append(f"  [{_mark(tasks)}] {tasks}")
    for label, path in [
        ("attempts", paths.all_trajectories_csv()),
        ("successes", paths.success_trajectories_json()),
    ]:
        lines.append(f"     [{_mark(path)}] {label}: {path}")

    lines += ["", "-- benchmark --"]
    # Every supervisor writes its own file for one (game, executor, model), so a single
    # path here would report 'missing' for a game that has four of the five on disk.
    for model in [paths.model_save_name, paths.finetuned_model_name]:
        for supervisor in BENCHMARK_SUPERVISORS:
            path = paths.benchmark_csv(paths.game, model, supervisor=supervisor)
            lines.append(f"  [{_mark(path)}] {path}")
    gbw = paths.gameboy_worlds_storage()
    lines.append(f"  GameBoyWorlds storage: {gbw or '(unreadable)'}")

    lines += ["", "-- output --", f"  {paths.debug_dir('check_paths')}"]
    log_info("\n".join(lines), obj["parameters"])
