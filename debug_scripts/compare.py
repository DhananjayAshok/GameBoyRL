"""
Paired comparison of two arbitrary benchmark CSVs.

Input
-----
Two CSVs written by any run_benchmark.py arm (baseline / info / plan):
    game, task, success, n_steps, n_invalid, subgoals_reached, all_subgoals, report,
    session_dirs
Unlike ``debug.py benchmark``, the two CSVs are given as **paths**, so any two runs can be
compared (base vs fine-tuned, two executors, two models, two prompting variants).

Tasks are paired on the ``task`` column and split into the four quadrants
(both pass, both fail, A only, B only); ``--n_examples`` episodes from each are rendered.

Frames
------
This command renders **no frames**. It used to reconstruct them by re-executing each
episode's recorded action sequence on the emulator, because nothing on disk carried the
frames a call actually saw. That is no longer true: every arm archives a
:class:`~execution.report.SupervisorReport` to ``<session_dir>/report.pkl.gz``, whose call
records hold their own images. Frame rendering therefore belongs to a reader of that
artifact, not to a parser of this CSV's rendered ``report`` text — and a replay was only ever
an illustration of the action sequence anyway, never the model's literal visual input.

What is left here is the part that needs no frames: the pairing, the quadrants, and the
paired significance test.

Output
------
<results_dir>/debug/<game>/compare/<label_a>__vs__<label_b>.md
"""

import os
import re

import click
import pandas as pd

from utils import log_info, log_warn, log_error
from debug_scripts import markdown as md
from debug_scripts.benchmark import _load
from utils.paths import Paths
from debug_scripts.stats import mcnemar_exact, wilson_str

QUADRANTS = [
    ("both_pass", "Both pass"),
    ("a_only", "{a} passes, {b} fails"),
    ("b_only", "{b} passes, {a} fails"),
    ("both_fail", "Both fail"),
]


def _episode_block(label: str, row) -> str:
    """One side of one paired episode: the numbers the CSV records, and where to look."""
    lines = [md.para(
        f"**{label}** — success: `{bool(row['success'])}` · steps: {row['n_steps']} · "
        f"invalid: {row['n_invalid']} · "
        f"subgoals: {row['subgoals_reached_n']}/{row['all_subgoals_n']}"
    )]
    sessions = row.get("session_dirs")
    if isinstance(sessions, str) and sessions.strip():
        lines.append(md.para(f"session: `{sessions}`"))
    return "\n".join(lines)


@click.command(name="compare")
@click.option("--csv_a", required=True, help="Path to the first benchmark CSV.")
@click.option("--csv_b", required=True, help="Path to the second benchmark CSV.")
@click.option("--label_a", default=None, help="Short name for A. Defaults to the CSV stem.")
@click.option("--label_b", default=None, help="Short name for B. Defaults to the CSV stem.")
@click.option("--n_examples", default=2, show_default=True,
              help="Episodes rendered per quadrant (0 = all).")
@click.pass_obj
def debug_compare(obj, csv_a, csv_b, label_a, label_b, n_examples):
    """Four-quadrant comparison of two benchmark CSVs."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], output_dir=obj["output_dir"], mode=obj["mode"],
    )
    report_dir = paths.debug_dir("compare")

    label_a = label_a or os.path.splitext(os.path.basename(csv_a))[0]
    label_b = label_b or os.path.splitext(os.path.basename(csv_b))[0]
    if label_a == label_b:
        label_a, label_b = f"{label_a} (A)", f"{label_b} (B)"

    a = _load(paths.require(csv_a, "benchmark"))
    b = _load(paths.require(csv_b, "benchmark"))
    log_info(f"[compare] {label_a}: {len(a)} tasks · {label_b}: {len(b)} tasks")

    games = set(a["game"]) | set(b["game"])
    if len(games) > 1:
        log_warn(f"[compare] CSVs span several games {sorted(games)}; "
                 f"the report is labelled '{paths.game}'.")

    shared = sorted(set(a["task"]) & set(b["task"]))
    if not shared:
        log_error("The two CSVs share no tasks — nothing to compare.", paths.parameters)
    only_a = sorted(set(a["task"]) - set(shared))
    only_b = sorted(set(b["task"]) - set(shared))
    fa = a[a["task"].isin(shared)].drop_duplicates("task").set_index("task").loc[shared]
    fb = b[b["task"].isin(shared)].drop_duplicates("task").set_index("task").loc[shared]

    quadrant_tasks = {
        "both_pass": [t for t in shared if fa.loc[t, "success"] and fb.loc[t, "success"]],
        "a_only": [t for t in shared if fa.loc[t, "success"] and not fb.loc[t, "success"]],
        "b_only": [t for t in shared if fb.loc[t, "success"] and not fa.loc[t, "success"]],
        "both_fail": [t for t in shared if not fa.loc[t, "success"] and not fb.loc[t, "success"]],
    }

    sections = []
    for key, title in QUADRANTS:
        tasks = quadrant_tasks[key]
        chosen = tasks[:n_examples] if n_examples else tasks
        blocks = [md.h2(f"{title.format(a=label_a, b=label_b)} — {len(tasks)} task(s)")]
        if not tasks:
            blocks.append(md.para("_(none)_"))
            sections.append("\n".join(blocks))
            continue
        blocks.append(md.para("Tasks: " + ", ".join(f"`{t}`" for t in tasks)))
        for task in chosen:
            blocks.append(md.h3(task))
            for label, frame in [(label_a, fa), (label_b, fb)]:
                blocks.append(_episode_block(label, frame.loc[task]))
        if n_examples and len(tasks) > len(chosen):
            blocks.append(md.para(
                f"_… {len(tasks) - len(chosen)} further task(s) in this quadrant "
                f"(--n_examples)_"
            ))
        sections.append("\n".join(blocks))

    n = len(shared)
    hits_a, hits_b = int(fa["success"].sum()), int(fb["success"].sum())
    summary = pd.DataFrame([
        {"run": label_a, "success": hits_a, "n": n, "success_%": hits_a / n * 100,
         "subgoal_frac_%": fa["subgoal_frac"].mean() * 100,
         "invalid_per_ep": fa["n_invalid"].mean(), "steps_per_ep": fa["n_steps"].mean()},
        {"run": label_b, "success": hits_b, "n": n, "success_%": hits_b / n * 100,
         "subgoal_frac_%": fb["subgoal_frac"].mean() * 100,
         "invalid_per_ep": fb["n_invalid"].mean(), "steps_per_ep": fb["n_steps"].mean()},
    ])

    blocks = [
        md.h1(f"Benchmark comparison — {paths.game}"),
        md.bullets([
            f"**A** = `{label_a}` — `{csv_a}`",
            f"**B** = `{label_b}` — `{csv_b}`",
        ]),
        md.h2("Pairing"),
        md.bullets([
            f"tasks compared: **{n}** (intersection)",
            f"only in A: **{len(only_a)}** {'(' + ', '.join(only_a[:10]) + ')' if only_a else ''}",
            f"only in B: **{len(only_b)}** {'(' + ', '.join(only_b[:10]) + ')' if only_b else ''}",
        ]),
        md.h2("Summary"),
        md.table(summary),
        md.bullets([
            f"A success: {wilson_str(hits_a, n)}",
            f"B success: {wilson_str(hits_b, n)}",
            f"both pass: **{len(quadrant_tasks['both_pass'])}** · "
            f"A only: **{len(quadrant_tasks['a_only'])}** · "
            f"B only: **{len(quadrant_tasks['b_only'])}** · "
            f"both fail: **{len(quadrant_tasks['both_fail'])}**",
            f"McNemar exact p on the "
            f"{len(quadrant_tasks['a_only'])}/{len(quadrant_tasks['b_only'])} discordant "
            f"pairs: **{mcnemar_exact(len(quadrant_tasks['a_only']), len(quadrant_tasks['b_only'])):.3f}**",
        ]),
        md.note(
            "No frames are rendered here. Each episode's frames live in the archived "
            "`report.pkl.gz` beside its video, under the `session_dirs` path recorded on the "
            "row — the supervisor report holds the images every call actually saw, which is "
            "strictly better than the emulator replay this command used to do."
        ),
    ] + sections

    stem_a = re.sub(r"[^\w.-]", "_", label_a)
    stem_b = re.sub(r"[^\w.-]", "_", label_b)
    path = md.write_report(os.path.join(report_dir, f"{stem_a}__vs__{stem_b}.md"), blocks)
    log_info(f"[compare] wrote {path}")
    print(path)
