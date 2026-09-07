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

#: Columns every game has, because they come from the environment's own subgoal metric
#: rather than from a per-game parser. `subgoals.n_completed` is badges won on any of the
#: Pokemon games -- Red's are boulder/cascade/..., Brown's marine/hail/..., Prism's
#: pyre/nature/... -- so the same column means the same thing across all of them.
TRACKED = [
    ("subgoals.n_completed", "objectives"),
    ("subgoals.next", "working on"),
    ("core.steps", "steps"),
]

#: Extra columns shown only when a game publishes them. Red's tracker exposes the starter
#: and a location count; the ROM hacks do not, and a column of dashes for them would imply
#: the agent failed at something rather than that the signal does not exist.
OPTIONAL = [
    ("pokemon_red_starter.current_starter", "starter"),
    ("pokemon_red_location.n_of_unique_locations", "locs"),
]


def _curve(run: dict) -> None:
    arm = "ABLATION" if run["init_kwargs"].get("use_notebook") is False else "control"
    print(f"--- {os.path.basename(run['_path'])}")
    print(f"    arm: {arm} | goal: {run['goal'][:60]}")
    print(f"    reached: {run['goal_achieved']} ({run['stop_reason']})")
    present = {k for t in run["tasks"] for k in (t.get("progress") or {})}
    columns = TRACKED + [(k, lbl) for k, lbl in OPTIONAL if k in present]
    print("    " + "task".ljust(6) + "".join(label.rjust(13) for _, label in columns))
    seen_any = False
    for task in run["tasks"]:
        prog = task.get("progress") or {}
        if not prog:
            continue
        seen_any = True
        cells = []
        for key, _ in columns:
            v = prog.get(key)
            # a badge name is long and only its tail distinguishes it: collect_boulder_badge
            # and collect_cascade_badge share a prefix, so truncating from the left would
            # render both as the same string.
            text = "-" if v is None else str(v).replace("collect_", "").replace("_badge", "")
            cells.append(text[-12:].rjust(13))
        print("    " + f"#{task['index']}".ljust(6) + "".join(cells))
    if not seen_any:
        print("    (no progress recorded — run predates progress snapshots)")
    # The end state is what a comparison actually turns on.
    last = next((t["progress"] for t in reversed(run["tasks"]) if t.get("progress")), {})
    if last:
        done = last.get("subgoals.completed") or []
        total = last.get("subgoals.n_total")
        bits = [f"objectives {len(done)}/{total}" if total else "objectives: not published"]
        if done:
            bits.append("won: " + ", ".join(g.replace("collect_", "") for g in done))
        if last.get("pokemon_red_starter.current_starter"):
            bits.append(f"starter {last['pokemon_red_starter.current_starter']}")
        if last.get("pokemon_red_location.unique_locations"):
            bits.append(f"locations {last['pokemon_red_location.unique_locations']}")
        print("    final: " + " | ".join(bits))
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
