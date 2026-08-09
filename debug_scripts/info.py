"""
Info-document diagnostics: is the built document any good, and does it produce good hints?

This vertical is almost entirely un-unit-testable — every stage is a model judgement, so
there is no assertion that catches a bad document, only a human reading the right rendering
of it. These two commands are the instrument.

`debug.py info` — offline, no VLM, no emulator
    Reads insights.jsonl and the whole merge/ tree.
      - the yield funnel (pairs in -> NONE rate -> leaves -> final entries)
      - benchmark init_state coverage, which gates --mode init_state entirely
      - the entry-count-per-round curve, the single most diagnostic plot: tracking the
        "never matched" reference line means the matcher never fires, a flat line means it
        fires on everything
      - insight drift, the anti-generality check: vague-phrase vs concrete-anchor ratio per
        round, plus one entry's Insights rendered side by side as the rounds rewrote it
      - the match audit, replayed from every matches.json
      - the final document itself

`debug.py info_hint` — needs a VLM and frames, never an emulator step
    Runs the InfoHintSupervisor's hint pipeline on sampled benchmark screens without playing
    the game, and reports the hint next to the full yes/no verdict list that produced it —
    so a bad hint can be attributed to relevance (wrong entries) or synthesis (right entries,
    bad hint), which are different prompt fixes and indistinguishable from the hint alone.

Input (all produced by scripts/vlm/build_info.sh)
------------------------------------------------
<info_dir>/insights.jsonl
<info_dir>/merge/round_<r>/<i>.{json,matches.json,meta.json}
<info_dir>/info.json

Output
------
<results_dir>/debug/<game>/info/report.md (and info_hint/report.md), plus figures beside them.
"""

import glob
import json
import os
import re

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from debug_scripts import markdown as md
from utils.paths import Paths
from execution.info_doc import TASK_SECTION, InfoDocument
from utils.paths import INFO_DOC_FILENAME
from utils import log_info, log_warn

# Phrases that signal an insight has drifted into unfalsifiable advice, and the concrete
# anchors that signal it has not. The ratio between them per merge round is the cheap
# automated read on generalisation-by-merge.
VAGUE_PHRASES = [
    "carefully", "make sure to", "pay attention", "as needed", "appropriately",
    "be aware", "in general", "if necessary", "keep in mind", "properly",
    "various", "some", "explore the area", "as appropriate",
]
CONCRETE_ANCHORS = [
    r"\bpress\b", r"\bbutton\b", r"\bA\b", r"\bB\b", r"\bstart\b", r"\bselect\b",
    r"\bup\b", r"\bdown\b", r"\bleft\b", r"\bright\b", r"\bnorth\b", r"\bsouth\b",
    r"\beast\b", r"\bwest\b", r"\bmenu\b", r"\bdoor\b", r"\bchest\b", r"\bNPC\b",
]
CONDITIONAL_MARKERS = [r"\bif\b", r"\bwhen\b", r"\bprovided\b", r"\bunless\b", r"\bonce\b"]


def _count_hits(text: str, patterns, regex: bool) -> int:
    lowered = text.lower()
    if not regex:
        return sum(lowered.count(p) for p in patterns)
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def _load_insights(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _round_dirs(info_dir: str) -> list[str]:
    root = os.path.join(info_dir, "merge")
    if not os.path.isdir(root):
        return []
    dirs = [d for d in glob.glob(os.path.join(root, "round_*")) if os.path.isdir(d)]
    return sorted(dirs, key=lambda d: int(os.path.basename(d).split("_")[1]))


def _round_documents(round_dir: str) -> list:
    docs = []
    # Node documents only: the sibling .matches.json / .meta.json are the merge's audit
    # trail, not documents, and would fail InfoDocument.from_dict.
    for doc_path in sorted(glob.glob(os.path.join(round_dir, "[0-9]*.json"))):
        if doc_path.endswith((".matches.json", ".meta.json")):
            continue
        with open(doc_path, "r") as handle:
            docs.append(InfoDocument.from_dict(json.load(handle)))
    return docs


def _figure_entry_counts(per_round: pd.DataFrame, n_leaves: int, out_path: str):
    """Entry count vs round, against the 'never matched' reference line."""
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(per_round["round"], per_round["task_entries"], marker="o",
              color="#4C78A8", label="task entries")
    axis.plot(per_round["round"], per_round["image_entries"], marker="o",
              color="#F58518", label="image entries")
    axis.axhline(n_leaves, ls="--", color="#888888",
                 label=f"never matched (= {n_leaves} leaves)")
    axis.set_xlabel("merge round")
    axis.set_ylabel("entries in the surviving documents")
    axis.set_title("Entry count per round\n(tracking the dashed line = matcher never fires)")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _figure_drift(per_round: pd.DataFrame, out_path: str):
    """Mean insight length and the vague:concrete ratio, per round."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.plot(per_round["round"], per_round["mean_insight_words"], marker="o", color="#54A24B")
    left.set_xlabel("merge round")
    left.set_ylabel("mean words per insight")
    left.set_title("Insight length\n(shrinking = smoothing away detail)")
    left.grid(alpha=0.3)

    right.plot(per_round["round"], per_round["vague_per_insight"], marker="o",
               color="#E45756", label="vague phrases")
    right.plot(per_round["round"], per_round["concrete_per_insight"], marker="o",
               color="#4C78A8", label="concrete anchors")
    right.set_xlabel("merge round")
    right.set_ylabel("hits per insight")
    right.set_title("Anti-generality check\n(vague rising over concrete = drift)")
    right.legend(frameon=False, fontsize=9)
    right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _document_stats(docs: list) -> dict:
    task_entries = sum(len(d.task_entries) for d in docs)
    image_entries = sum(len(d.image_entries) for d in docs)
    insights, words, vague, concrete = [], 0, 0, 0
    for doc in docs:
        for entry in doc.task_entries + doc.image_entries:
            for insight in entry.insights:
                insights.append(insight)
                words += len(insight.split())
                vague += _count_hits(insight, VAGUE_PHRASES, regex=False)
                concrete += _count_hits(insight, CONCRETE_ANCHORS, regex=True)
    n = max(len(insights), 1)
    return {
        "task_entries": task_entries,
        "image_entries": image_entries,
        "n_insights": len(insights),
        "mean_insight_words": words / n,
        "vague_per_insight": vague / n,
        "concrete_per_insight": concrete / n,
    }


@click.command()
@click.option("--model_name", required=True, help="VLM that built the document.")
@click.option("--source", default="attempt", show_default=True,
              type=click.Choice(["attempt", "curiosity"]),
              help="Which vertical's info dir to read; selects the paths.py accessor.")
@click.option("--max_entries", default=40, show_default=True,
              help="Cap on entries rendered in full (0 = no cap).")
@click.pass_obj
def debug_info(obj, model_name, source, max_entries):
    """Info-document build diagnostics: funnel, coverage, merge curve, drift, match audit."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], model_name=model_name, output_dir=obj["output_dir"],
        mode=obj["mode"],
    )
    report_dir = paths.debug_dir("info")
    images_dir = paths.debug_dir("info", "images")

    info_dir = paths.source_info_dir(source)
    insights_path = paths.require(os.path.join(info_dir, "insights.jsonl"), "insights")
    rows = _load_insights(insights_path)
    log_info(f"[info] {len(rows)} stage-A leaves from {insights_path}")

    leaf_docs = [InfoDocument.from_dict(row["document"]) for row in rows]
    leaf_stats = _document_stats(leaf_docs)

    # --- Funnel -----------------------------------------------------------
    # The number of pairs that went in is the annotation json's length; insights.jsonl only
    # holds the survivors, so the difference is the NONE rate.
    n_pairs = None
    for candidate in (paths.success_trajectories_json() if source == "attempt"
                      else paths.curiosity_annotation(),):
        if os.path.exists(candidate):
            with open(candidate) as handle:
                n_pairs = len(json.load(handle))

    funnel = [
        f"pairs available upstream: **{n_pairs}**" if n_pairs
        else "pairs available upstream: _(annotation json absent)_",
        f"leaves extracted (non-NONE): **{len(rows)}**",
    ]
    if n_pairs:
        dropped = n_pairs - len(rows)
        funnel.append(f"dropped as NONE / unparseable: **{dropped}** "
                      f"({100.0 * dropped / max(n_pairs, 1):.0f}%)")
    funnel.append(f"leaf entries: **{leaf_stats['task_entries']}** task / "
                  f"**{leaf_stats['image_entries']}** image")
    funnel.append(f"leaf insights: **{leaf_stats['n_insights']}**")

    # --- Benchmark init_state coverage (gates --mode init_state) ----------
    # Reading the benchmark task file here is fine: this is a read-only diagnostic and no
    # benchmark text ever enters the document.
    coverage_block = []
    try:
        from gameboy_worlds import get_benchmark_tasks

        bench = get_benchmark_tasks(game=paths.game)
        counts = {}
        for row in rows:
            counts[row.get("init_state")] = counts.get(row.get("init_state"), 0) + 1
        coverage = (
            bench.groupby("init_state").size().reset_index(name="benchmark_tasks")
        )
        coverage["stage_a_records"] = coverage["init_state"].map(lambda s: counts.get(s, 0))
        coverage = coverage.sort_values("stage_a_records")
        n_zero = int((coverage["stage_a_records"] == 0).sum())
        coverage_block = [
            md.para(
                f"**{len(coverage) - n_zero}/{len(coverage)}** benchmark init states have at "
                f"least one stage-A record. Zero-coverage states are exactly the episodes "
                f"`--mode init_state` will run hintless on."
            ),
            md.table(coverage),
        ]
        if n_zero:
            coverage_block.append(md.warn(
                f"{n_zero} benchmark init state(s) have no records — `--mode init_state` "
                f"cannot hint on them, and its score will be diluted accordingly."
            ))
    except Exception as error:
        coverage_block = [md.warn(f"Could not compute benchmark coverage: {error}")]

    # --- Merge tree -------------------------------------------------------
    round_dirs = _round_dirs(info_dir)
    per_round_rows = [dict(round=-1, **leaf_stats)]      # round -1 == the leaves themselves
    match_records = []
    for round_dir in round_dirs:
        round_idx = int(os.path.basename(round_dir).split("_")[1])
        docs = _round_documents(round_dir)
        if docs:
            per_round_rows.append(dict(round=round_idx, **_document_stats(docs)))
        for matches_path in sorted(glob.glob(os.path.join(round_dir, "*.matches.json"))):
            with open(matches_path) as handle:
                matches = json.load(handle)
            node = os.path.basename(matches_path).split(".")[0]
            for section, entries in matches.items():
                for record in entries:
                    match_records.append({
                        "round": round_idx,
                        "node": node,
                        "section": "task" if section == TASK_SECTION else "image",
                        "candidate": record.get("candidate"),
                        "matched": record.get("matched") or "— (appended fresh)",
                        "reason": (record.get("reason") or "")[:160],
                    })

    per_round = pd.DataFrame(per_round_rows).sort_values("round")
    merge_blocks = []
    if len(round_dirs):
        counts_fig = os.path.join(images_dir, "entry_counts.png")
        drift_fig = os.path.join(images_dir, "insight_drift.png")
        _figure_entry_counts(per_round, leaf_stats["task_entries"] + leaf_stats["image_entries"],
                             counts_fig)
        _figure_drift(per_round, drift_fig)
        merge_blocks = [
            md.h2("Merge tree"),
            md.para(f"{len(round_dirs)} round(s) under `{os.path.join(info_dir, 'merge')}`. "
                    "Round -1 is the unmerged leaves."),
            md.img("entry counts per round", counts_fig, report_dir),
            md.img("insight drift", drift_fig, report_dir),
            md.table(per_round.round(2)),
        ]
        if len(match_records):
            match_frame = pd.DataFrame(match_records)
            n_matched = int((match_frame["matched"] != "— (appended fresh)").sum())
            merge_blocks += [
                md.h2("Match audit"),
                md.para(f"**{n_matched}/{len(match_frame)}** candidate entries matched an "
                        "existing entry; the rest were appended verbatim. A misfiled entry is "
                        "found here."),
                md.table(match_frame.head(200)),
            ]
    else:
        merge_blocks = [md.h2("Merge tree"),
                        md.warn("No `merge/` tree yet — stage B has not been run. "
                                "`--mode init_state` works from the leaves alone.")]

    # --- The document itself ---------------------------------------------
    doc_blocks = []
    doc_path = os.path.join(info_dir, INFO_DOC_FILENAME)
    if os.path.exists(doc_path):
        with open(doc_path) as handle:
            document = InfoDocument.from_dict(json.load(handle))
        summary = pd.DataFrame([
            {"section": "task", "category": e.category,
             "examples": len(e.examples), "insights": len(e.insights),
             "init_states": ", ".join(e.init_states)}
            for e in document.task_entries
        ] + [
            {"section": "image", "category": e.category,
             "examples": len(e.examples), "insights": len(e.insights),
             "init_states": ", ".join(e.init_states)}
            for e in document.image_entries
        ])
        doc_blocks = [
            md.h2("Final document"),
            md.para(f"`{doc_path}` — **{len(document.task_entries)}** task / "
                    f"**{len(document.image_entries)}** image entries."),
            md.table(summary),
        ]
        entries = document.task_entries + document.image_entries
        if max_entries:
            entries = entries[:max_entries]
        for entry in entries:
            body = (f"**Description:** {entry.description}\n\n"
                    f"**Examples:**\n{md.bullets(entry.examples)}\n"
                    f"**Insights:**\n{md.bullets(entry.insights)}")
            doc_blocks.append(md.details(f"{entry.category}", body))
    else:
        doc_blocks = [md.h2("Final document"),
                      md.warn(f"No `{INFO_DOC_FILENAME}` at {doc_path} — stage B has not been run.")]

    blocks = [
        md.h1(f"Info document — {paths.game} / {paths.model_save_name} / source={source}"),
        md.para(f"Source: `{info_dir}`"),
        md.h2("Yield funnel"),
        md.bullets(funnel),
        md.note("If most pairs return NONE, or the final document has a handful of entries, "
                "nothing downstream can work and the fix is the stage-A prompt, not a "
                "benchmark run."),
        md.h2("Benchmark init_state coverage"),
        *coverage_block,
        *merge_blocks,
        *doc_blocks,
    ]
    out = md.write_report(os.path.join(report_dir, "report.md"), blocks)
    log_info(f"[info] wrote {out}")


@click.command()
@click.option("--model_name", required=True, help="VLM that built the document.")
@click.option("--source", default="attempt", show_default=True,
              type=click.Choice(["attempt", "curiosity"]))
@click.option("--hint_mode", default="retrieval", show_default=True,
              type=click.Choice(["retrieval", "init_state", "both"]),
              help="Which selection path to exercise. 'both' writes both hints for the same "
                   "screen, so any difference is purely which insights were selected.")
@click.option("--hint_vlm_model", default=None, help="Defaults to --model_name.")
@click.option("--hint_vlm_kind", default="openai", show_default=True)
@click.option("--n_tasks", default=5, show_default=True,
              help="Benchmark tasks to spot-check (0 = all).")
@click.option("--max_concurrency", default=8, show_default=True)
@click.pass_obj
def debug_info_hint(obj, model_name, source, hint_mode, hint_vlm_model, hint_vlm_kind,
                    n_tasks, max_concurrency):
    """Hint spot-check: run the hint pipeline on benchmark screens without playing the game."""
    import time

    from gameboy_worlds import get_benchmark_tasks, get_test_environment

    from debug_scripts.frames import to_pil
    from execution.info_doc import load_document
    from execution.registry import AVAILABLE_EXECUTORS
    from execution.supervisors import InfoHintSupervisor

    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], model_name=model_name, output_dir=obj["output_dir"],
        mode=obj["mode"],
    )
    report_dir = paths.debug_dir("info_hint")
    images_dir = paths.debug_dir("info_hint", "images")
    info_dir = paths.source_info_dir(source)

    modes = ["retrieval", "init_state"] if hint_mode == "both" else [hint_mode]
    documents, insight_rows = None, None
    if "retrieval" in modes:
        doc_path = paths.require(os.path.join(info_dir, INFO_DOC_FILENAME), "info")
        # No source= argument: the document records its own provenance, and load_document
        # copies the label onto every entry.
        documents = [load_document(doc_path, parameters=obj["parameters"])]
    if "init_state" in modes:
        insights_path = paths.require(os.path.join(info_dir, "insights.jsonl"), "insights")
        insight_rows = _load_insights(insights_path)

    bench = get_benchmark_tasks(game=paths.game)
    if n_tasks:
        bench = bench.head(n_tasks)

    records = []
    for i, row in bench.iterrows():
        environment = get_test_environment(row=row, controller_variant="low_level",
                                           headless=True, save_video=False,
                                           session_name=f"debug_info_hint/{i}", max_steps=1)
        try:
            frame = environment.get_info()["core"]["current_frame"]
            screen_path = os.path.join(images_dir, f"screen_{i}.png")
            to_pil(frame).save(screen_path)

            for mode in modes:
                supervisor = InfoHintSupervisor(
                    task=row["task"],
                    executor_class=AVAILABLE_EXECUTORS["simple"],
                    env=environment,
                    game=row["game"],
                    max_steps=1,
                    max_tool_calls=0,
                    documents=documents,
                    insight_rows=insight_rows,
                    mode=mode,
                    init_state=row["init_state"],
                    hint_vlm_model=hint_vlm_model or model_name,
                    hint_vlm_kind=hint_vlm_kind,
                    max_concurrency=max_concurrency,
                    parameters=obj["parameters"],
                )
                started = time.time()
                hint = supervisor.write_hint()
                records.append({
                    "index": i,
                    "task": row["task"],
                    "init_state": row["init_state"],
                    "mode": mode,
                    "hint": hint,
                    "screen": screen_path,
                    "selection": list(supervisor.selection_log),
                    # Same ids the benchmark CSV stores, so a hint seen here and a hint seen
                    # in a results file are traceable to evidence the same way.
                    "selected_ids": list(supervisor.selected_ids),
                    "seconds": time.time() - started,
                    "n_calls": len(supervisor.selection_log) + 1,
                })
                log_info(f"[info_hint] {mode} / task {i}: "
                         f"{'NO HINT' if hint is None else hint[:80]}")
        except Exception as error:
            log_warn(f"[info_hint] task {i} failed: {error}")
        finally:
            environment.close()

    if not records:
        log_warn("[info_hint] no records produced; nothing to report.")
        return

    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ("selection", "selected_ids")}
                          for r in records])
    per_mode = []
    for mode, group in frame.groupby("mode"):
        n = len(group)
        n_hint = int(group["hint"].notna().sum())
        conditional = sum(
            1 for h in group["hint"].dropna()
            if _count_hits(h, CONDITIONAL_MARKERS, regex=True) > 0
        )
        per_mode.append({
            "mode": mode,
            "screens": n,
            "hinted": n_hint,
            "NO HINT %": round(100.0 * (n - n_hint) / max(n, 1), 1),
            # The executor treats every hint as reliable, so hedging in the hint text is the
            # only brake on a confidently wrong one. A low number here is the failure signature.
            "conditional %": round(100.0 * conditional / max(n_hint, 1), 1),
            "mean calls": round(group["n_calls"].mean(), 1),
            "mean seconds": round(group["seconds"].mean(), 1),
        })

    fire_counts = {}
    for record in records:
        for verdict in record["selection"]:
            key = (verdict["kind"], verdict["category"])
            stats = fire_counts.setdefault(key, {"seen": 0, "fired": 0})
            stats["seen"] += 1
            stats["fired"] += int(verdict["relevant"])
    fire_frame = pd.DataFrame([
        {"kind": kind, "category": category, "judged": s["seen"], "fired": s["fired"],
         "fire rate %": round(100.0 * s["fired"] / max(s["seen"], 1), 1)}
        for (kind, category), s in fire_counts.items()
    ]).sort_values("fire rate %", ascending=False) if fire_counts else pd.DataFrame()

    blocks = [
        md.h1(f"Hint spot-check — {paths.game} / {paths.model_save_name} / source={source}"),
        md.para(f"Source: `{info_dir}` — no emulator steps were taken; each screen is the "
                "opening frame of a benchmark task."),
        md.h2("Summary"),
        md.table(pd.DataFrame(per_mode)),
        md.note("A hint that is not conditional cannot fail safely: the executor's hint block "
                "presents every hint as reliable, so `if/when` scoping in the hint text is the "
                "only guard against a confidently wrong one."),
    ]
    if len(fire_frame):
        blocks += [
            md.h2("Fire rate per entry"),
            md.para("Entries that fire on every screen are too generic; entries that never "
                    "fire are dead weight paying a relevance call every episode."),
            md.table(fire_frame),
        ]

    blocks.append(md.h2("Hints"))
    for record in records:
        blocks.append(md.h3(f"[{record['mode']}] {record['task']}"))
        blocks.append(md.para(f"init_state: `{record['init_state']}`"))
        blocks.append(md.img(f"screen {record['index']}", record["screen"], report_dir))
        blocks.append(md.para(f"**Hint:** {record['hint'] or '_NO HINT_'}"))
        if record["selected_ids"]:
            blocks.append(md.para("**Synthesised from:** "
                                  + ", ".join(f"`{i}`" for i in record["selected_ids"])))
        if record["selection"]:
            verdicts = pd.DataFrame(record["selection"])[
                ["kind", "category", "relevant", "reason"]
            ]
            blocks.append(md.details(
                f"relevance verdicts ({int(verdicts['relevant'].sum())}/{len(verdicts)} fired)",
                md.table(verdicts),
            ))

    out = md.write_report(os.path.join(report_dir, "report.md"), blocks)
    log_info(f"[info_hint] wrote {out}")
