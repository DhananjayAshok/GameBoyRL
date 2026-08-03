"""
SFT dataset diagnostics: what the model was actually trained on.

Input (produced by scripts/vlm/create_dataset.sh)
-------------------------------------------------
<practice_dir>/train_dataset.csv, validation_dataset.csv
    task_string, input, output, image, score
    `image` is a comma-joined list of absolute paths into <practice_dir>/images/.

Images are **copied** into the report's own ``images/<leg>/`` dir and linked from there.
create_dataset already wrote the JPEGs, but under ``storage_dir`` (``/project2/...``), which
is outside the VS Code workspace root — markdown preview will not load out-of-workspace
images, so a relative link straight to them renders blank. Copying the sampled frames
in-workspace (like every other report) is what makes them show. Only the sampled rows are
copied, so the cost is bounded by ``--n_samples``.

Output
------
<results_dir>/debug/<game>/dataset/report_<leg>.md
"""

import os
import re
import shutil

import click
import pandas as pd

from utils import log_info, log_warn, log_error, parse_action_line
from debug_scripts import markdown as md
from debug_scripts.paths import Paths
from debug_scripts.stats import gini

# create_dataset now drops any call with an unparseable or illegal `Action:` (its
# VALID_ACTIONS is the source of truth), so this report no longer audits action validity —
# every row's action is valid by construction. It still shows the action *distribution*.
PROMPT_BLOCKS = {
    "Recent actions": "history section (HistoryAwareExecutor)",
    "[ERROR]": "parse-failure feedback",
    "Tool result:": "tool output",
    "Available tool calls": "tool menu (absent when max_tool_calls=0)",
    "[HINT": "hint block (stripped by create_dataset)",
    "[STEP_INFO": "step-info block (stripped by create_dataset)",
}


def _action_counts(frame: pd.DataFrame) -> pd.Series:
    """Distribution of parsed `Action:` values across training targets. create_dataset
    guarantees every row has a valid, parseable action, so this is a coverage view, not a
    validity check."""
    return frame["output"].map(parse_action_line).dropna().str.strip().str.upper().value_counts()


@click.command(name="dataset")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.option("--leg", default="both", show_default=True,
              type=click.Choice(["curiosity", "zeroshot", "both"]),
              help="Which data-collection leg's dataset to report. 'both' writes one report "
                   "per leg. The merged dataset is not a leg — it is the concatenation of "
                   "these two and carries a `source` column that reconstructs the split.")
@click.option("--extra", default="none", show_default=True,
              help="Which proposal variant, within the zeroshot leg only.")
@click.option("--n_samples", default=50, show_default=True,
              help="Random rows to show, split across train and validation.")
@click.option("--seed", default=0, show_default=True, help="Seed for the sample.")
@click.pass_obj
def debug_dataset(obj, model_name, leg, extra, n_samples, seed):
    """Show random train/validation rows verbatim, with an automatic target audit."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], model_name=model_name, output_dir=obj["output_dir"], mode=obj["mode"],
    )
    legs = ["curiosity", "zeroshot"] if leg == "both" else [leg]
    written = [_dataset_report(paths, obj, name, extra, n_samples, seed) for name in legs]
    for path in written:
        print(path)


def _dataset_report(paths, obj, leg, extra, n_samples, seed):
    report_dir = paths.debug_dir("dataset")
    # Per-leg so curiosity/zeroshot copies never collide on a shared basename.
    images_dir = paths.debug_dir("dataset", "images", leg)
    practice_dir = paths.leg_dir(leg, extra)

    train_path = paths.require(paths.train_csv(practice_dir), "dataset")
    val_path = paths.require(paths.validation_csv(practice_dir), "dataset")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    log_info(f"[dataset/{leg}] {len(train)} train / {len(val)} validation rows")

    if n_samples > len(train) + len(val):
        log_error(
            f"--n_samples {n_samples} exceeds the {len(train) + len(val)} rows available.",
            paths.parameters,
        )

    counts = _action_counts(train)
    total = len(train)
    rows_per_task = train["task_string"].value_counts()

    action_table = pd.DataFrame({
        "action": counts.index,
        "rows": counts.values,
        "share_%": counts.values / max(total, 1) * 100,
    }).head(25)

    block_rows = []
    for needle, meaning in PROMPT_BLOCKS.items():
        present = int(train["input"].str.contains(needle, regex=False, na=False).sum())
        block_rows.append({
            "block": needle,
            "meaning": meaning,
            "train_rows": present,
            "share_%": present / max(total, 1) * 100,
        })
    blocks_table = pd.DataFrame(block_rows)

    # Leakage: exact duplicates across the split, plus shared images and task strings.
    train_pairs = set(zip(train["input"].astype(str), train["output"].astype(str)))
    val_pairs = set(zip(val["input"].astype(str), val["output"].astype(str)))
    dup_pairs = len(train_pairs & val_pairs)
    train_images = set(", ".join(train["image"].dropna().astype(str)).split(", "))
    val_images = set(", ".join(val["image"].dropna().astype(str)).split(", "))
    shared_images = len(train_images & val_images) - (1 if "" in (train_images & val_images) else 0)
    shared_tasks = len(set(train["task_string"]) & set(val["task_string"]))

    # Sample rows proportionally from both splits.
    n_val = max(1, round(n_samples * len(val) / max(len(train) + len(val), 1)))
    n_train = max(0, n_samples - n_val)
    picked = pd.concat([
        train.sample(n=min(n_train, len(train)), random_state=seed).assign(split="train"),
        val.sample(n=min(n_val, len(val)), random_state=seed).assign(split="validation"),
    ]).sample(frac=1.0, random_state=seed)

    sample_blocks = []
    missing_images = 0
    for i, (_, row) in enumerate(picked.iterrows(), 1):
        parts = [md.h3(f"{i}. [{row['split']}] {row['task_string']}")]
        image_field = row.get("image")
        if isinstance(image_field, str) and image_field.strip():
            for image_path in [p.strip() for p in image_field.split(",") if p.strip()]:
                if os.path.exists(image_path):
                    # Copy in-workspace so the link renders in markdown preview (the source
                    # lives under /project2, outside the workspace root). Basenames are
                    # unique per call, so this is idempotent across re-runs.
                    local = os.path.join(images_dir, os.path.basename(image_path))
                    if obj["overwrite"] or not os.path.exists(local):
                        shutil.copyfile(image_path, local)
                    parts.append(md.img("frame", local, report_dir))
                else:
                    missing_images += 1
                    parts.append(md.para(f"_(image missing: `{image_path}`)_"))
        parts.append(md.para("**input**"))
        parts.append(md.code(row["input"]))
        parts.append(md.para("**output**"))
        parts.append(md.code(row["output"]))
        sample_blocks.append("\n".join(parts))

    if missing_images:
        log_warn(f"[dataset] {missing_images} referenced images are missing on disk")

    report_blocks = [
        md.h1(f"SFT dataset — {paths.game} / {paths.model_save_name} / leg={leg}"),
        md.para(f"Leg: **{leg}** · source: `{practice_dir}`"),
        md.h2("Size"),
        md.bullets([
            f"train rows: **{len(train)}** over **{train['task_string'].nunique()}** task strings",
            f"validation rows: **{len(val)}** over **{val['task_string'].nunique()}** task strings",
            f"unique training targets (`output`): **{train['output'].nunique()}**",
            f"rows-per-task Gini: **{gini(rows_per_task.values):.3f}** "
            f"(top 10 tasks hold {rows_per_task.head(10).sum() / max(total, 1) * 100:.1f}% of rows)",
        ]),
        md.h2("Action distribution (train)"),
        md.para(
            "create_dataset drops any call with an unparseable or illegal `Action:`, so every "
            "training target here is a valid environment action by construction."
        ),
        md.table(action_table),
        md.h2("Prompt blocks present in training inputs"),
        md.table(blocks_table),
        md.para(
            "Compare these shares against the prompts the executor actually builds at benchmark "
            "time (see the benchmark report's `report` column). A block at 0% here but present at "
            "inference — or vice versa — is a train/inference distribution mismatch."
        ),
        md.h2("Leakage checks"),
        md.bullets([
            f"exact `(input, output)` pairs in **both** train and validation: **{dup_pairs}**",
            f"image files referenced by **both** splits: **{shared_images}**",
            f"task strings in **both** splits: **{shared_tasks}** "
            "(expected to be low — create_dataset reserves the last paraphrase for validation)",
        ]),
        md.h2(f"{len(picked)} random samples"),
        md.para(f"Seed {seed}. Images are the JPEGs `create_dataset` already wrote, linked in place."),
        "\n\n".join(sample_blocks),
    ]

    report_path = md.write_report(os.path.join(report_dir, f"report_{leg}.md"), report_blocks)
    log_info(f"[dataset/{leg}] wrote {report_path}")
    return report_path
