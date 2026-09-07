#!/usr/bin/env python
"""
Progress curve for a long-horizon strategist run.

    python scripts/debug/progress_curve.py [--game pokemon_red] [--last 2]

A whole-game goal is binary and reads "not reached" for a very long time, which cannot
distinguish an arm that explored six locations from one that never left the first room.
This prints what actually moved, per task, so two arms can be compared on distance
travelled rather than on a zero they both share.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import click

from utils import load_parameters

#: Signals worth a column. Ordered roughly by how much progress they represent.
TRACKED = [
    ("badges", "badges"),
    ("pokemon_red_starter.current_starter", "starter"),
    ("pokemon_red_location.n_of_unique_locations", "locs"),
    ("pokemon_red_location.n_walk_steps", "walked"),
    ("core.steps", "steps"),
]


def _curve(run: dict) -> None:
    arm = "ABLATION" if run["init_kwargs"].get("use_notebook") is False else "control"
    print(f"--- {os.path.basename(run['_path'])}")
    print(f"    arm: {arm} | goal: {run['goal'][:60]}")
    print(f"    reached: {run['goal_achieved']} ({run['stop_reason']})")
    header = "    " + "task".ljust(6) + "".join(label.rjust(10) for _, label in TRACKED)
    print(header)
    seen_any = False
    for task in run["tasks"]:
        prog = task.get("progress") or {}
        if not prog:
            continue
        seen_any = True
        cells = []
        for key, _ in TRACKED:
            v = prog.get(key)
            cells.append(("-" if v is None else str(v))[:10].rjust(10))
        print("    " + f"#{task['index']}".ljust(6) + "".join(cells))
    if not seen_any:
        print("    (no progress recorded — run predates progress snapshots)")
    # The end state is what a comparison actually turns on.
    last = next((t["progress"] for t in reversed(run["tasks"]) if t.get("progress")), {})
    if last:
        locs = last.get("pokemon_red_location.unique_locations")
        print(f"    final: badges={last.get('badges')} starter={last.get('pokemon_red_starter.current_starter')} "
              f"locations={locs}")
    print()


@click.command()
@click.option("--game", default="pokemon_red", type=str)
@click.option("--last", default=2, type=int)
def main(game, last):
    """Print per-task progress curves for recent runs."""
    parameters = load_parameters()
    pattern = os.path.join(parameters["storage_dir"], "strategist", game, "*.json")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)[:last]
    if not paths:
        raise click.ClickException(f"No records under {pattern}")
    for path in paths:
        with open(path) as handle:
            run = json.load(handle)
        run["_path"] = path
        _curve(run)


if __name__ == "__main__":
    main()
