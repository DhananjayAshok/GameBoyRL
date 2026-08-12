"""
Benchmark diagnostics: per-episode frame-by-frame trajectories, and a paired
base-vs-fine-tuned diff.

Input
-----
<results_dir>/benchmark/<game>/<executor>_<model>.csv
    game, task, success, n_steps, n_invalid, subgoals_reached, all_subgoals, report,
    session_dirs
`success` is the environment's own verdict (termination_reason == "terminated"), not a
VLM judge — this is the only ground-truth success signal in the pipeline.

<session_dir>/report.pkl.gz    (session_dir from the row's `session_dirs`)
    The archived :class:`~execution.report.SupervisorReport`. **This is what the report is
    built from.** Its ``event_log`` interleaves the supervisor's own calls with the full
    report of every executor leg, and each call record carries the images it saw, its
    prompt, the raw response and the steps it produced.

This used to parse the CSV's rendered `report` column with a regex over box-drawing
characters, because that string was the only durable per-episode record. It cost three
things: no frames at all, only the *last* step per call (so a `sequence` executor lost every
action but one), and no way to see the supervisor's own reasoning. Reading the archive
instead fixes all three. The `report` column is still written and is still useful for
grepping; nothing here parses it.

Output
------
<results_dir>/debug/<game>/benchmark/episodes_<model>.md   (one per model)
<results_dir>/debug/<game>/benchmark/comparison.md
Frames go to <storage_dir>/tmp/debug_frames/<game>/benchmark/<model>/<task>/ — on storage,
not beside the markdown, because there are thousands of them per report.
"""

import ast
import gzip
import json
import os
import pickle

import click
import pandas as pd

from execution.report import (EnvironmentStepRecord, SupervisorVLMCallRecord,
                              _step_summary)
from utils import log_error, log_info, log_warn, parse_action_line
from benchmark_scripts.common import REPORT_FILENAME
from debug_scripts import markdown as md
from debug_scripts.frames import to_pil
from utils.paths import Paths
from debug_scripts.stats import mcnemar_exact, wilson, wilson_str


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


# ---------------------------------------------------------------------------
# The archive
# ---------------------------------------------------------------------------


def _session_dir(row):
    """The emulator session directory recorded on one CSV row, or None."""
    raw = row.get("session_dirs")
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        dirs = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return dirs[0] if dirs else None


def _load_report(row):
    """The archived SupervisorReport for one episode, or None if it has none."""
    session = _session_dir(row)
    if session is None:
        return None
    path = os.path.join(session, REPORT_FILENAME)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)["report"]


def _require_archives(frame: pd.DataFrame, csv_path: str, parameters: dict) -> dict:
    """Load every episode's report, refusing a CSV that predates the archive.

    Refused rather than degraded: a report that silently renders without frames looks like a
    run that made no calls, and comparing one of those against a complete one is worse than
    getting an error. A *few* missing archives are tolerated with a warning, since
    ``save_report`` deliberately swallows its own failures so an unwritable archive cannot
    cost an already-paid-for episode.
    """
    if "session_dirs" not in frame.columns:
        log_error(
            f"{csv_path} has no session_dirs column, so its episodes cannot be located. It "
            "was written before the report archive existed — re-run the benchmark to get one.",
            parameters,
        )
    reports = {}
    for task, row in frame.set_index("task").iterrows():
        report = _load_report(row)
        if report is not None:
            reports[task] = report
    if not reports:
        log_error(
            f"No report.pkl.gz found for any episode in {csv_path}. The session directories "
            "it names hold no archive, so this CSV predates report archiving — re-run the "
            "benchmark.",
            parameters,
        )
    if len(reports) < len(frame):
        log_warn(f"[benchmark] {len(frame) - len(reports)} of {len(frame)} episodes have no "
                 f"archive; they are rendered from their CSV row alone.")
    return reports


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")[:60]


def _storage_link(report_dir: str, name: str, target: str, parameters: dict) -> str:
    """A symlink inside the report dir pointing at *target* on storage.

    Frames and videos live on ``/project2`` while the markdown lives under ``results/`` on
    the home filesystem — eight ``../`` levels apart. That relative path resolves fine on
    disk, but every markdown renderer refuses to load an image from outside the workspace
    root, so the images silently do not appear. One symlink per report keeps the links short
    and inside the tree while the bytes stay on storage.

    Refuses to replace a real directory: that would be deleting whatever a previous run or a
    person put there. A stale symlink pointing somewhere else is just repointed.
    """
    link = os.path.join(report_dir, name)
    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(target):
            return link
        os.unlink(link)
    elif os.path.exists(link):
        log_error(
            f"{link} exists and is not a symlink, so the report cannot point it at "
            f"{target}. Move it aside.",
            parameters,
        )
    os.symlink(target, link)
    return link


def _via_link(path: str, target_root: str, link_root: str) -> str:
    """Rewrite a storage path to go through the report's symlink instead."""
    return os.path.join(link_root, os.path.relpath(path, target_root))


def _save_frame(frame, directory: str, name: str, overwrite: bool):
    """One numpy frame to PNG, returning its path (or None if it could not be rendered)."""
    path = os.path.join(directory, f"{name}.png")
    if os.path.exists(path) and not overwrite:
        return path
    try:
        to_pil(frame).save(path)
    except Exception as error:  # noqa: BLE001 - a bad frame must not kill the report
        log_warn(f"[benchmark] could not write {path}: {error}")
        return None
    return path


def _executor_call_blocks(call, index: int, frames_dir: str, report_dir: str,
                          prefix: str, overwrite: bool) -> list:
    """One executor VLM call: what it saw, what it was asked, what it said, what it did."""
    lines = [md.para(f"**call {index}** · tag `{call.tag}`")]

    for i, image in enumerate(call.images):
        path = _save_frame(image, frames_dir, f"{prefix}_call{index}_saw{i}", overwrite)
        if path:
            lines.append(md.img(f"call {index} input {i}", path, report_dir))
    if not call.images:
        lines.append(md.para("_(no images on this call)_"))

    lines.append(md.details(f"call {index} prompt", md.code(call.prompt)))
    lines.append(md.para("output"))
    lines.append(md.code(call.response))

    action = parse_action_line(call.response)
    lines.append(md.para(f"parsed action: `{action if action else 'none'}`"))

    # Every step, not just the last: one call can own several (SequencePlannerExecutor), and
    # the old text parser kept only the final one.
    if call.steps:
        lines.append(md.code("\n".join(
            f"{i + 1:>3}  {_step_summary(step)}" for i, step in enumerate(call.steps)
        )))
    else:
        lines.append(md.para("_(this call took no steps)_"))

    env_steps = [s for s in call.steps if isinstance(s, EnvironmentStepRecord)]
    if env_steps:
        path = _save_frame(env_steps[-1].frame_after, frames_dir,
                           f"{prefix}_call{index}_after", overwrite)
        if path:
            lines.append(md.img(f"call {index} next frame", path, report_dir))
    return lines


def _event_blocks(report, frames_dir: str, report_dir: str, overwrite: bool) -> list:
    """The whole event log in order: supervisor calls and executor legs, interleaved."""
    lines = []
    leg = 0
    for position, event in enumerate(report.event_log):
        if isinstance(event, SupervisorVLMCallRecord):
            lines.append(md.h3(f"supervisor · {event.stage}"))
            for i, image in enumerate(event.images):
                path = _save_frame(image, frames_dir, f"sup{position}_saw{i}", overwrite)
                if path:
                    lines.append(md.img(f"supervisor {event.stage} input {i}", path, report_dir))
            lines.append(md.details(f"{event.stage} prompt", md.code(event.prompt)))
            lines.append(md.para("output"))
            lines.append(md.code(str(event.response)))
        else:
            leg += 1
            hint = (event.init_kwargs or {}).get("hint")
            lines.append(md.h3(f"executor leg {leg} · task: {event.task!r}"))
            lines.append(md.bullets([
                f"termination: `{event.termination_reason}`",
                f"step budget: **{event.max_steps}**",
                f"hint: {f'`{hint}`' if hint else '_none_'}",
                f"VLM calls: **{len(event.vlm_call_log)}**",
            ]))
            for index, call in enumerate(event.vlm_call_log, 1):
                lines += _executor_call_blocks(
                    call, index, frames_dir, report_dir, f"leg{leg}", overwrite)
    return lines


def _episode_section(row, report, paths, bench_game, model, report_dir,
                     frames_root: str, video_roots, overwrite: bool) -> str:
    """One task: its outcome, its video, then every call frame by frame."""
    subgoals = _as_list(row["subgoals_reached"])
    all_subgoals = _as_list(row["all_subgoals"])
    task = row["task"]

    blocks = [
        md.h2(str(task)),
        md.bullets([
            f"success: **{row['success']}**",
            f"steps: **{row['n_steps']}** · invalid: **{row['n_invalid']}**",
            f"subgoals: **{len(subgoals)}/{len(all_subgoals)}** "
            f"({', '.join(subgoals) if subgoals else 'none reached'})",
        ]),
    ]

    video = paths.video_path(bench_game, model, task)
    if video and video_roots is not None:
        target = _via_link(video, *video_roots)
        blocks.append(md.para(f"video: {md.link(os.path.basename(video), target, report_dir)}"))
    elif video:
        blocks.append(md.para(f"video: `{video}`"))
    else:
        blocks.append(md.para("_(no video recorded)_"))

    if report is None:
        blocks.append(md.warn("No archived report for this episode — nothing to walk."))
        return "\n".join(blocks)

    blocks.append(md.bullets([
        f"supervisor: `{report.supervisor_name}`",
        f"events: **{len(report.event_log)}** "
        f"({len(report.supervisor_calls)} supervisor call(s), "
        f"{len(report.executor_reports)} executor leg(s))",
    ]))
    frames_dir = os.path.join(frames_root, _slug(model), _slug(task))
    os.makedirs(frames_dir, exist_ok=True)
    blocks += _event_blocks(report, frames_dir, report_dir, overwrite)
    return "\n".join(blocks)


@click.command(name="benchmark")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.option("--compare_model", default="none", show_default=True,
              help="Fine-tuned served-model name. 'none' derives "
                   "<model_save_name>-<game>-<run_name> as serve_and_benchmark.sh does.")
@click.option("--bench_game", default="none", show_default=True,
              help="Game whose benchmark CSVs to read. 'none' uses --game.")
@click.option("--max_episodes", default=0, show_default=True,
              help="Cap episodes rendered in episodes_*.md (0 = all).")
@click.pass_obj
def debug_benchmark(obj, model_name, compare_model, bench_game, max_episodes):
    """Per-episode frame-by-frame trajectories plus a paired base-vs-fine-tuned comparison."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], model_name=model_name, output_dir=obj["output_dir"],
        mode=obj["mode"],
    )
    report_dir = paths.debug_dir("benchmark")
    overwrite = obj["overwrite"]
    game = paths.game if bench_game == "none" else bench_game

    # Frames and videos are on storage; the markdown is not. Link both into the report dir so
    # the image and video targets stay inside the tree — see _storage_link.
    frames_root = _storage_link(
        report_dir, "frames", paths.debug_frames_dir("benchmark"), paths.parameters)
    gbw = paths.gameboy_worlds_storage()
    video_roots = None
    if gbw is not None:
        sessions_root = os.path.join(gbw, "sessions", game)
        if os.path.isdir(sessions_root):
            video_roots = (sessions_root,
                           _storage_link(report_dir, "videos", sessions_root,
                                         paths.parameters))
    base_model = paths.model_save_name
    ft_model = paths.finetuned_model_name if compare_model == "none" else compare_model

    base_path = paths.require(paths.benchmark_csv(game, base_model), "benchmark")
    base = _load(base_path)
    base_reports = _require_archives(base, base_path, paths.parameters)
    log_info(f"[benchmark] base: {len(base)} tasks from {base_path} "
             f"({len(base_reports)} archived)")

    ft_path = paths.benchmark_csv(game, ft_model)
    finetuned, ft_reports = None, {}
    if os.path.exists(ft_path):
        finetuned = _load(ft_path)
        ft_reports = _require_archives(finetuned, ft_path, paths.parameters)
        log_info(f"[benchmark] finetuned: {len(finetuned)} tasks from {ft_path}")
    else:
        log_warn(f"[benchmark] no fine-tuned CSV at {ft_path} — comparison will be skipped.")

    written = []

    # ---------------- (a) per-episode trajectories ----------------
    for model, frame, reports in [(base_model, base, base_reports),
                                  (ft_model, finetuned, ft_reports)]:
        if frame is None:
            continue
        rows = frame.head(max_episodes) if max_episodes else frame
        sections = [
            _episode_section(row, reports.get(row["task"]), paths, game, model, report_dir,
                             frames_root, video_roots, overwrite)
            for _, row in rows.iterrows()
        ]
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
                "Every call below is read from the archived supervisor report beside that "
                f"episode's video, not reconstructed from text. Frames are written to "
                f"`{frames_root}` — on storage, since a full report is thousands of PNGs."
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

        def _step_lines(report) -> str:
            """Every step the episode took, in order, from its archive."""
            if report is None:
                return "_(no archive)_"
            steps = [_step_summary(step)
                     for leg in report.executor_reports
                     for call in leg.vlm_call_log
                     for step in call.steps]
            return md.code("\n".join(f"{i + 1:>3}  {s}" for i, s in enumerate(steps)))

        def _diff_sections(tasks, left_model, right_model):
            out = []
            for task in tasks:
                left_report, right_report = base_reports.get(task), ft_reports.get(task)
                lines = [
                    md.h3(task),
                    md.table(pd.DataFrame([
                        {"model": left_model, "success": bool(b.loc[task, "success"]),
                         "steps": b.loc[task, "n_steps"], "invalid": b.loc[task, "n_invalid"],
                         "subgoals": f"{b.loc[task, 'subgoals_reached_n']}/{b.loc[task, 'all_subgoals_n']}"},
                        {"model": right_model, "success": bool(f.loc[task, "success"]),
                         "steps": f.loc[task, "n_steps"], "invalid": f.loc[task, "n_invalid"],
                         "subgoals": f"{f.loc[task, 'subgoals_reached_n']}/{f.loc[task, 'all_subgoals_n']}"},
                    ])),
                ]
                for model, report in [(left_model, left_report), (right_model, right_report)]:
                    video = paths.video_path(game, model, task)
                    link = ""
                    if video and video_roots is not None:
                        target = _via_link(video, *video_roots)
                        link = f" · video: {md.link('mp4', target, report_dir)}"
                    lines.append(md.para(f"**{model}** steps{link}"))
                    lines.append(_step_lines(report))
                    calls = [c for leg in (report.executor_reports if report else [])
                             for c in leg.vlm_call_log]
                    if calls:
                        lines.append(md.details(f"{model} — final VLM output",
                                                md.code(calls[-1].response)))
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
