"""
Scaffolding check: resolve every artifact path for the given identity and report which
exist. Run this first when pointing the debug tools at a new game/run — it tells you
which stages have artifacts before any report tries to read them.
"""

import os

import click

from debug_scripts.paths import Paths, EXTRA_SUFFIXES


def _mark(path: str) -> str:
    return "OK  " if os.path.exists(path) else "MISS"


@click.command(name="hello")
@click.option("--model_name", default=None,
              help="Full VLM name (e.g. google/gemma-4-31b-it). Omit to check curiosity paths only.")
@click.pass_obj
def hello(obj, model_name):
    """Print every derived path with an exists/missing marker."""
    paths = Paths(
        parameters=obj["parameters"],
        game=obj["game"],
        run_name=obj["run_name"],
        executor=obj["executor"],
        model_name=model_name,
        output_dir=obj["output_dir"],
    )

    print(f"game       : {paths.game}")
    print(f"run_name   : {paths.run_name}")
    print(f"executor   : {paths.executor}")
    print(f"model_name : {model_name or '(not given)'}")
    print(f"storage_dir: {paths.storage_dir}")
    print(f"results_dir: {paths.results_dir}")
    print()

    print("-- curiosity --")
    grouped = paths.grouped_dir()
    print(f"  [{_mark(grouped)}] {grouped}")
    if os.path.exists(grouped):
        states = sorted(d for d in os.listdir(grouped) if os.path.exists(paths.grouped_file(d)))
        print(f"         {len(states)} init_state(s) with grouped trajectories")

    if model_name is None:
        print("\n(model_name not given — skipping model-keyed stages)")
        return

    print("\n-- proposal / attempt / practice --")
    print(f"  [{_mark(paths.curiosity_annotation())}] {paths.curiosity_annotation()}")
    for extra in EXTRA_SUFFIXES:
        tasks = paths.tasks_file(extra)
        if not os.path.exists(tasks):
            continue
        print(f"  [{_mark(tasks)}] {tasks}")
        for label, path in [
            ("attempts", paths.all_trajectories_csv(extra)),
            ("guidance", paths.guidance_json(extra)),
            ("practice", paths.practice_results_csv(extra)),
            ("clean   ", paths.clean_decisions_csv(extra)),
            ("train   ", paths.train_csv(extra)),
            ("val     ", paths.validation_csv(extra)),
        ]:
            print(f"     [{_mark(path)}] {label}: {path}")

    print("\n-- benchmark --")
    for model in [paths.model_save_name, paths.finetuned_model_name]:
        path = paths.benchmark_csv(paths.game, model)
        print(f"  [{_mark(path)}] {path}")
    gbw = paths.gameboy_worlds_storage()
    print(f"  GameBoyWorlds storage: {gbw or '(unreadable)'}")

    print(f"\n-- output --\n  {paths.debug_dir('hello')}")
