"""
Benchmark diagnostics: per-episode trajectories, and a paired base-vs-fine-tuned diff.

Input (produced by scripts/benchmark.sh / scripts/pipeline/serve_and_benchmark.sh)
----------------------------------------------------------------------------------
<results_dir>/benchmark/<game>/<executor>_<model>.csv
    game, task, success, n_resets, n_steps, n_invalid, subgoals_reached, all_subgoals, report
`success` is the environment's own verdict (termination_reason == "terminated"), not a
VLM judge — this is the only ground-truth success signal in the pipeline.

The `report` column holds the full rendered trajectory: every prompt, every VLM output and
every step outcome, as written by ExecutorReport.__str__. That text is per-model and is
what this report renders.

Why not the frame PNGs
----------------------
ExecutorReport.__str__ writes per-call PNGs to
<results_dir>/benchmark/<game>/<ExecutorClassName>/<task>/ — keyed on the executor *class*,
not the model — and rmtree's the directory first. A base run and a fine-tuned run therefore
overwrite each other's frames and whatever is on disk belongs to whichever ran last. Those
PNGs are unusable for a model comparison and are not linked. The recorded **videos** are
model-specific (session dir includes the served model name) and are linked instead.

Output
------
<results_dir>/debug/<game>/benchmark/episodes_<model>.md   (one per model)
<results_dir>/debug/<game>/benchmark/comparison.md
"""

import ast
import os
import re

import click
import pandas as pd

from utils import log_info, log_warn
from debug_scripts import markdown as md
from debug_scripts.paths import Paths
from debug_scripts.stats import mcnemar_exact, wilson, wilson_str

CALL_HEADER = re.compile(r"^\s*┌─ \[([A-Z_]+)\] \(call (\d+)\)")
CALL_FOOTER = re.compile(r"^\s*└─+")
PROMPT_MARK = "| Prompt:"
OUTPUT_MARK = "│ VLM output:"
STEP_MARK = "│ → "
INDENT = "  │   "


def parse_report(report: str) -> list[dict]:
    """
    Parse an ExecutorReport rendering back into per-call records.

    :return: list of ``{"call", "tag", "prompt", "output", "outcome"}``.
    """
    if not isinstance(report, str) or not report.strip():
        return []
    calls, current, section = [], None, None
    for line in report.splitlines():
        header = CALL_HEADER.match(line)
        if header:
            if current:
                calls.append(current)
            current = {"call": int(header.group(2)), "tag": header.group(1).lower(),
                       "prompt": [], "output": [], "outcome": ""}
            section = None
            continue
        if current is None:
            continue
        if CALL_FOOTER.match(line):
            calls.append(current)
            current, section = None, None
            continue
        if line.strip().startswith(PROMPT_MARK.strip()):
            section = "prompt"
            continue
        if line.strip().startswith(OUTPUT_MARK.strip()):
            section = "output"
            continue
        if STEP_MARK in line:
            current["outcome"] = line.split(STEP_MARK, 1)[1].strip()
            section = None
            continue
        if section:
            current[section].append(line[len(INDENT):] if line.startswith(INDENT) else line.strip())
    if current:
        calls.append(current)
    for call in calls:
        call["prompt"] = "\n".join(call["prompt"]).strip()
        call["output"] = "\n".join(call["output"]).strip()
    return calls


def _as_list(value):
    """subgoals_reached / all_subgoals are stringified python lists in the CSV."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _load(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["success"] = frame["success"].astype(bool)
    frame["subgoals_reached_n"] = frame["subgoals_reached"].map(lambda v: len(_as_list(v)))
    frame["all_subgoals_n"] = frame["all_subgoals"].map(lambda v: len(_as_list(v)))
    frame["subgoal_frac"] = frame.apply(
        lambda r: r["subgoals_reached_n"] / r["all_subgoals_n"] if r["all_subgoals_n"] else 0.0,
        axis=1,
    )
    return frame


def _termination(report: str) -> str:
    """Coarse outcome class from a rendered report — what ended the episode."""
    calls = parse_report(report)
    if not calls:
        return "no calls"
    outcomes = [c["outcome"] for c in calls if c["outcome"]]
    if not outcomes:
        return "no steps"
    invalid = sum(1 for o in outcomes if o.startswith("INVALID"))
    if invalid == len(outcomes):
        return "all invalid"
    return "ran to budget"


def _episode_section(row, calls, paths, bench_game, model, report_dir, max_calls):
    subgoals = _as_list(row["subgoals_reached"])
    all_subgoals = _as_list(row["all_subgoals"])
    head = [
        md.h2(f"{row['task']}"),
        md.bullets([
            f"success: **{row['success']}**",
            f"steps: **{row['n_steps']}** · invalid: **{row['n_invalid']}** · "
            f"resets: **{row['n_resets']}**",
            f"subgoals: **{len(subgoals)}/{len(all_subgoals)}** "
            f"({', '.join(subgoals) if subgoals else 'none reached'})",
            f"VLM calls recorded: **{len(calls)}**",
        ]),
    ]
    video = paths.video_path(bench_game, model, row["task"])
    if video:
        head.append(md.para(f"video: {md.link(os.path.basename(video), video, report_dir)}"))

    body = []
    for call in calls[:max_calls] if max_calls else calls:
        body.append(f"**call {call['call']}** [{call['tag']}] → `{call['outcome'] or 'n/a'}`\n")
        body.append(md.details("prompt", md.code(call["prompt"])))
        body.append(md.para("output"))
        body.append(md.code(call["output"]))
    if max_calls and len(calls) > max_calls:
        body.append(md.para(f"_… {len(calls) - max_calls} further calls omitted (--max_calls)_"))
    return "\n".join(head + body)


@click.command(name="benchmark")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.option("--compare_model", default="none", show_default=True,
              help="Fine-tuned served-model name. 'none' derives "
                   "<model_save_name>-<game>-<run_name> as serve_and_benchmark.sh does.")
@click.option("--bench_game", default="none", show_default=True,
              help="Game whose benchmark CSVs to read. 'none' uses --game.")
@click.option("--max_episodes", default=0, show_default=True,
              help="Cap episodes rendered in episodes_*.md (0 = all).")
@click.option("--max_calls", default=0, show_default=True,
              help="Cap calls rendered per episode (0 = all).")
@click.pass_obj
def debug_benchmark(obj, model_name, compare_model, bench_game, max_episodes, max_calls):
    """Per-episode trajectories plus a paired base-vs-fine-tuned comparison."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], model_name=model_name, output_dir=obj["output_dir"],
    )
    report_dir = paths.debug_dir("benchmark")
    game = paths.game if bench_game == "none" else bench_game
    base_model = paths.model_save_name
    ft_model = paths.finetuned_model_name if compare_model == "none" else compare_model

    base_path = paths.require(paths.benchmark_csv(game, base_model), "benchmark")
    base = _load(base_path)
    log_info(f"[benchmark] base: {len(base)} tasks from {base_path}")

    ft_path = paths.benchmark_csv(game, ft_model)
    finetuned = None
    if os.path.exists(ft_path):
        finetuned = _load(ft_path)
        log_info(f"[benchmark] finetuned: {len(finetuned)} tasks from {ft_path}")
    else:
        log_warn(f"[benchmark] no fine-tuned CSV at {ft_path} — comparison will be skipped.")

    written = []

    # ---------------- (a) per-episode trajectories ----------------
    for model, frame in [(base_model, base), (ft_model, finetuned)]:
        if frame is None:
            continue
        rows = frame.head(max_episodes) if max_episodes else frame
        sections = []
        for _, row in rows.iterrows():
            calls = parse_report(row["report"])
            sections.append(
                _episode_section(row, calls, paths, game, model, report_dir, max_calls)
            )
        n_success = int(frame["success"].sum())
        blocks = [
            md.h1(f"Benchmark episodes — {game} / {paths.executor} / {model}"),
            md.para(f"Source: `{paths.benchmark_csv(game, model)}`"),
            md.bullets([
                f"tasks: **{len(frame)}**",
                f"success: {wilson_str(n_success, len(frame))}",
                f"mean subgoal fraction: **{frame['subgoal_frac'].mean() * 100:.1f}%**",
                f"mean invalid actions per episode: **{frame['n_invalid'].mean():.2f}**",
            ]),
            md.note(
                "Trajectories are reconstructed from the CSV's `report` column, which is written "
                "per model. The per-call PNGs under "
                f"`{os.path.join(paths.results_dir, 'benchmark', game)}/<ExecutorClass>/` are keyed "
                "on the executor class rather than the model and are overwritten by whichever run "
                "finished last, so they are deliberately not linked here."
            ),
            "\n\n".join(sections),
        ]
        path = md.write_report(os.path.join(report_dir, f"episodes_{model}.md"), blocks)
        log_info(f"[benchmark] wrote {path}")
        written.append(path)

    # ---------------- (b) paired comparison ----------------
    if finetuned is not None:
        shared = sorted(set(base["task"]) & set(finetuned["task"]))
        dropped_base = sorted(set(base["task"]) - set(shared))
        dropped_ft = sorted(set(finetuned["task"]) - set(shared))
        b = base[base["task"].isin(shared)].set_index("task").loc[shared]
        f = finetuned[finetuned["task"].isin(shared)].set_index("task").loc[shared]

        base_only = [t for t in shared if b.loc[t, "success"] and not f.loc[t, "success"]]
        ft_only = [t for t in shared if f.loc[t, "success"] and not b.loc[t, "success"]]
        both = [t for t in shared if b.loc[t, "success"] and f.loc[t, "success"]]
        neither = [t for t in shared if not b.loc[t, "success"] and not f.loc[t, "success"]]

        n = len(shared)
        base_hits, ft_hits = int(b["success"].sum()), int(f["success"].sum())
        p_value = mcnemar_exact(len(base_only), len(ft_only))
        base_lo, base_hi = wilson(base_hits, n)
        ft_lo, ft_hi = wilson(ft_hits, n)

        summary = pd.DataFrame([
            {"model": base_model, "success": base_hits, "n": n,
             "success_%": base_hits / n * 100, "ci_low_%": base_lo * 100, "ci_high_%": base_hi * 100,
             "subgoal_frac_%": b["subgoal_frac"].mean() * 100,
             "invalid_per_ep": b["n_invalid"].mean(), "steps_per_ep": b["n_steps"].mean()},
            {"model": ft_model, "success": ft_hits, "n": n,
             "success_%": ft_hits / n * 100, "ci_low_%": ft_lo * 100, "ci_high_%": ft_hi * 100,
             "subgoal_frac_%": f["subgoal_frac"].mean() * 100,
             "invalid_per_ep": f["n_invalid"].mean(), "steps_per_ep": f["n_steps"].mean()},
        ])

        def _diff_sections(tasks, left_model, right_model):
            out = []
            for task in tasks:
                left_calls = parse_report(b.loc[task, "report"])
                right_calls = parse_report(f.loc[task, "report"])
                left_actions = [c["outcome"] for c in left_calls if c["outcome"]]
                right_actions = [c["outcome"] for c in right_calls if c["outcome"]]
                lines = [
                    md.h3(task),
                    md.table(pd.DataFrame([
                        {"model": left_model, "success": bool(b.loc[task, "success"]),
                         "steps": b.loc[task, "n_steps"], "invalid": b.loc[task, "n_invalid"],
                         "subgoals": f"{b.loc[task, 'subgoals_reached_n']}/{b.loc[task, 'all_subgoals_n']}",
                         "calls": len(left_calls)},
                        {"model": right_model, "success": bool(f.loc[task, "success"]),
                         "steps": f.loc[task, "n_steps"], "invalid": f.loc[task, "n_invalid"],
                         "subgoals": f"{f.loc[task, 'subgoals_reached_n']}/{f.loc[task, 'all_subgoals_n']}",
                         "calls": len(right_calls)},
                    ])),
                ]
                for model, calls, actions in [
                    (left_model, left_calls, left_actions),
                    (right_model, right_calls, right_actions),
                ]:
                    video = paths.video_path(game, model, task)
                    link = f" · video: {md.link('mp4', video, report_dir)}" if video else ""
                    lines.append(md.para(f"**{model}** step outcomes{link}"))
                    lines.append(md.code("\n".join(f"{i + 1:>3}  {a}" for i, a in enumerate(actions))))
                    if calls:
                        lines.append(md.details(
                            f"{model} — final VLM output",
                            md.code(calls[-1]["output"]),
                        ))
                out.append("\n".join(lines))
            return "\n\n".join(out) if out else md.para("_(none)_")

        blocks = [
            md.h1(f"Benchmark comparison — {game} / {paths.executor}"),
            md.para(f"`{base_model}` (base) vs `{ft_model}` (fine-tuned)"),
            md.h2("Pairing"),
            md.bullets([
                f"tasks compared: **{n}** (intersection of both CSVs)",
                f"dropped, only in base: **{len(dropped_base)}** "
                f"{'(' + ', '.join(dropped_base[:10]) + ')' if dropped_base else ''}",
                f"dropped, only in fine-tuned: **{len(dropped_ft)}** "
                f"{'(' + ', '.join(dropped_ft[:10]) + ')' if dropped_ft else ''}",
            ]),
            md.h2("Summary"),
            md.table(summary),
            md.bullets([
                f"solved by base only: **{len(base_only)}**",
                f"solved by fine-tuned only: **{len(ft_only)}**",
                f"solved by both: **{len(both)}** · by neither: **{len(neither)}**",
                f"McNemar exact p on the {len(base_only)}/{len(ft_only)} discordant pairs: "
                f"**{p_value:.3f}**",
            ]),
            md.para(
                "Success is binary over a few dozen tasks, so the confidence intervals overlap "
                "easily; `subgoal_frac_%` uses the partial progress already recorded in every row "
                "and is the lower-variance signal to judge by. `invalid_per_ep` tests whether "
                "unparseable or non-executable training targets leaked into behaviour."
            ),
            md.h2(f"Solved by base only ({len(base_only)})"),
            md.para("Regressions introduced by fine-tuning — the debugging shortlist."),
            _diff_sections(base_only, base_model, ft_model),
            md.h2(f"Solved by fine-tuned only ({len(ft_only)})"),
            _diff_sections(ft_only, base_model, ft_model),
            md.h2("Both / neither"),
            md.table(pd.DataFrame({
                "outcome": ["both solved"] * len(both) + ["neither solved"] * len(neither),
                "task": both + neither,
            })),
        ]
        path = md.write_report(os.path.join(report_dir, "comparison.md"), blocks)
        log_info(f"[benchmark] wrote {path}")
        written.append(path)

    for path in written:
        print(path)
