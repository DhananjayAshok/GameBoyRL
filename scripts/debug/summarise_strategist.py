#!/usr/bin/env python
"""
Summarise strategist runs: what was planned, what stuck, and what the notebook bought.

    python scripts/debug/summarise_strategist.py [--game pokemon_red] [--last 4]

Reads the JSON records written by ``run_strategist.py`` and prints one block per run plus a
control-vs-ablation comparison when both are present. Kept as a script rather than folded
into the runner because it is read after the fact, often over several runs at once, and a
runner that also reported across runs would have to know about files it did not write.
"""

from __future__ import annotations

import glob
import json
import os
import sys

# Run as a script from the repo root, so the repo is not already on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import click

from execution.strategists.notebook import COMPRESS_THRESHOLD
from utils import load_parameters


def _load(path: str) -> dict:
    with open(path) as handle:
        blob = json.load(handle)
    blob["_path"] = path
    return blob


def _describe(run: dict) -> None:
    """One run, in the detail needed to judge whether it made progress."""
    notebook = run.get("notebook") or {}
    ablated = run["init_kwargs"].get("use_notebook") is False
    print(f"--- {os.path.basename(run['_path'])}")
    print(f"    arm            : {'ABLATION (no notebook)' if ablated else 'control (notebook)'}")
    print(f"    goal           : {run['goal']}")
    print(f"    reached        : {run['goal_achieved']}  ({run['stop_reason']})")
    print(f"    planned tasks  : {run['n_planned_tasks']}  "
          f"({run['n_successful_tasks']} ended agent_done)")
    print(f"    emulator steps : {run['n_steps']}")
    print(f"    invalid actions: {run['n_invalid']}")
    print(f"    map facts      : {len(notebook.get('world_map', []))}")
    print(f"    ledger         : {notebook.get('ledger', {})}")
    print(f"    strategist toks: in={run['strategist_input_tokens']} "
          f"out={run['strategist_output_tokens']}")
    # The accordion. Reported explicitly because a fold is invisible in the task list —
    # the log simply gets shorter — and because it went unexercised for six runs without
    # anything saying so.
    folded = notebook.get("folded_count", 0)
    summary = notebook.get("compressed_history")
    if summary:
        print(f"    compression    : FIRED, folded {folded} attempts into "
              f"{len(summary)} chars, {len(notebook.get('attempts', []))} held verbatim")
        print(f"                     \"{summary[:150]}\"")
    else:
        print(f"    compression    : not reached (needs >= {COMPRESS_THRESHOLD} attempts, "
              f"this run made {len(notebook.get('attempts', []))})")
    print("    tasks issued:")
    for task in run["tasks"]:
        kind = "VERIFY" if task["is_verification"] else " TASK "
        status = "ok    " if task["success"] else "failed"
        print(f"      [{kind} #{task['index']:>2} {status}] {task['task']}")
        if task["lesson"]:
            print(f"                        lesson: {task['lesson']}")
    # Repetition is the failure this layer exists to prevent, so it is reported outright
    # rather than left for the reader to notice.
    planned = [t["task"].strip().lower() for t in run["tasks"] if not t["is_verification"]]
    repeats = len(planned) - len(set(planned))
    print(f"    verbatim repeats: {repeats}"
          + ("  <-- the notebook is not preventing re-issues" if repeats else ""))
    print()


@click.command()
@click.option("--game", default="pokemon_red", type=str)
@click.option("--last", default=4, type=int, help="Most recent N records to read.")
def main(game, last):
    """Print a summary of recent strategist runs."""
    parameters = load_parameters()
    pattern = os.path.join(parameters["storage_dir"], "strategist", game, "*.json")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)[:last]
    if not paths:
        raise click.ClickException(f"No strategist records found under {pattern}")

    runs = [_load(path) for path in paths]
    print(f"=== {len(runs)} strategist run(s) on {game} ===\n")
    for run in runs:
        _describe(run)

    control = next((r for r in runs if r["init_kwargs"].get("use_notebook") is not False), None)
    ablation = next((r for r in runs if r["init_kwargs"].get("use_notebook") is False), None)
    if control and ablation:
        print("=== control vs ablation ===")
        print(f"{'metric':<22}{'control':>12}{'ablation':>12}")
        for label, key in [("goal reached", "goal_achieved"),
                           ("planned tasks", "n_planned_tasks"),
                           ("tasks ended agent_done", "n_successful_tasks"),
                           ("emulator steps", "n_steps"),
                           ("invalid actions", "n_invalid")]:
            print(f"{label:<22}{str(control[key]):>12}{str(ablation[key]):>12}")
        for label, run in (("control", control), ("ablation", ablation)):
            planned = [t["task"].strip().lower() for t in run["tasks"] if not t["is_verification"]]
            print(f"{'verbatim repeats (' + label + ')':<22}"
                  f"{len(planned) - len(set(planned)):>12}")


if __name__ == "__main__":
    main()
