#!/usr/bin/env python
"""
Is the executor doing what the planner asked, and is anything changing on screen?

A playthrough that ends 0/8 says the run failed but not where. These four numbers say
where, and each one was added because a real run was ambiguous without it:

``adherence``      Content-word overlap between the planner's task and the executor's own
                   stated reasoning. Low means the executor is not working on the task it
                   was given -- a planner problem is upstream of everything else, so this
                   is checked first.
``dead``           Share of steps whose predecessor did nothing. High means the executor
                   is walking into a wall.
``repeat``         Of those, how often it pressed the SAME thing again. This is the number
                   that mattered: on Red it was 86% while the prompt was explicitly telling
                   the model it was stuck, which is what ruled out "it needs more context"
                   and pointed at the executor itself.
``effective``      Share of steps that moved the screen at all. The budget is 175 steps per
                   task; how many of them did anything is the honest progress denominator.

Read straight off the saved reports, so it needs no GPU and no emulator.

    python scripts/debug/executor_health.py --game pokemon_prism --last 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

VALID_ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"}

#: Words carrying no information about *what* a task is, so they must not count toward
#: adherence -- "the go to and" overlaps with every task ever written.
STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "is", "are", "was", "be",
    "it", "its", "this", "that", "with", "for", "from", "by", "as", "i", "you", "your",
    "my", "we", "should", "would", "will", "can", "could", "then", "there", "here", "so",
    "if", "but", "not", "no", "any", "some", "all", "into", "out", "up", "down", "left",
    "right", "next", "current", "currently", "now", "step", "action", "press", "button",
    "screen", "game", "goal", "task", "need", "want", "try", "trying", "move", "moving",
}


def _words(text: str) -> set:
    """Content words, crudely stemmed so 'walking' matches 'walk'."""
    out = set()
    for word in re.findall(r"[a-z]+", (text or "").lower()):
        if len(word) < 3 or word in STOP_WORDS:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        out.add(word)
    return out


def parse_leg(text: str) -> Dict:
    """One executor leg: its task, and every decision inside it."""
    task = ""
    match = re.search(r"EXECUTOR LEG \d+: '(.*?)'", text)
    if match:
        task = match.group(1)

    decisions: List[Dict] = []
    recent: List[Tuple[str, bool]] = []
    in_recent = False
    in_output = False
    reasoning: List[str] = []

    for line in text.splitlines():
        stripped = line.strip().lstrip("|│ ").strip()

        if stripped.startswith("Recent actions"):
            recent, in_recent = [], True
            continue
        if in_recent:
            entry = re.match(r"^([A-Z]+)(\s*\[no change\])?$", stripped)
            if entry:
                recent.append((entry.group(1), bool(entry.group(2))))
                continue
            in_recent = False

        if "VLM output:" in line:
            in_output, reasoning = True, []
            continue

        if in_output and stripped.startswith("Reasoning:"):
            reasoning.append(stripped.split("Reasoning:", 1)[1])
            continue

        if stripped.startswith("Action:"):
            action = stripped.split("Action:", 1)[1].strip().upper().strip('.<>"')
            # The prompt template carries a literal 'Action: <one environment action>'.
            # Only a real action closes a decision; anything else leaves state alone so
            # the template line cannot consume the history block meant for the next one.
            if action in VALID_ACTIONS:
                decisions.append({"action": action,
                                  "reasoning": " ".join(reasoning),
                                  "recent": list(recent)})
                recent, in_output, reasoning = [], False, []
    return {"task": task, "decisions": decisions}


def score(legs: List[Dict]) -> Dict:
    total = dead = repeated = effective = 0
    adherence: List[float] = []
    actions: Counter = Counter()

    for leg in legs:
        task_words = _words(leg["task"])
        for decision in leg["decisions"]:
            actions[decision["action"]] += 1
            if not decision["recent"]:
                continue
            total += 1
            previous, unchanged = decision["recent"][-1]
            if unchanged:
                dead += 1
                if decision["action"] == previous:
                    repeated += 1
            else:
                effective += 1
            if task_words and decision["reasoning"]:
                shared = task_words & _words(decision["reasoning"])
                adherence.append(len(shared) / len(task_words))

    def pct(part, whole):
        return 100.0 * part / whole if whole else float("nan")

    return {
        "steps": total,
        "dead_pct": pct(dead, total),
        "repeat_pct": pct(repeated, dead),
        "effective_pct": pct(effective, total),
        "adherence_pct": 100.0 * sum(adherence) / len(adherence) if adherence else float("nan"),
        "top_actions": actions.most_common(4),
        "n_actions": sum(actions.values()),
    }


def load(path: str) -> Dict:
    with open(path) as handle:
        record = json.load(handle)
    legs, objectives = [], None
    for task in record.get("tasks", []):
        report = task.get("report") or ""
        for chunk in re.split(r"(?==+ EXECUTOR LEG )", report):
            if "EXECUTOR LEG" in chunk:
                legs.append(parse_leg(chunk))
        progress = task.get("progress") or {}
        if "subgoals.n_completed" in progress:
            objectives = (progress.get("subgoals.n_completed"),
                          progress.get("subgoals.n_total"))
    result = score(legs)
    result["objectives"] = objectives
    result["n_tasks"] = len(record.get("tasks", []))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", default="pokemon_red")
    parser.add_argument("--storage", default=os.path.expanduser("~/gameboy_storage"))
    parser.add_argument("--last", type=int, default=4, help="Most recent N runs.")
    args = parser.parse_args()

    pattern = os.path.join(args.storage, "strategist", args.game, "*.json")
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)[-args.last:]
    if not paths:
        # The runner also writes under the repo's own storage dir.
        paths = sorted(glob.glob(os.path.join("storage", "strategist", args.game, "*.json")),
                       key=os.path.getmtime)[-args.last:]
    if not paths:
        raise SystemExit(f"no runs found for {args.game}")

    print(f"{'run':34} {'tasks':>5} {'steps':>6} {'adhere':>7} {'dead':>6} "
          f"{'repeat':>7} {'effect':>7} {'obj':>6}")
    for path in paths:
        row = load(path)
        name = os.path.basename(path)[:34]
        objectives = ("-" if row["objectives"] is None
                      else f"{row['objectives'][0]}/{row['objectives'][1]}")
        print(f"{name:34} {row['n_tasks']:>5} {row['steps']:>6} "
              f"{row['adherence_pct']:6.1f}% {row['dead_pct']:5.1f}% "
              f"{row['repeat_pct']:6.1f}% {row['effective_pct']:6.1f}% {objectives:>6}")
        print(f"{'':34} actions: {row['top_actions']}")


if __name__ == "__main__":
    main()
