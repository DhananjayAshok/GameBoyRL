"""
SFT dataset diagnostics: what the model was actually trained on.

Input (produced by scripts/vlm/create_dataset.sh)
-------------------------------------------------
<practice_dir>/train_dataset.csv, validation_dataset.csv
    task_string, input, output, image, score
    `image` is a comma-joined list of absolute paths into <practice_dir>/images/.

Images are **reused**, never re-rendered — create_dataset already wrote the JPEGs and the
CSV points straight at them. They are linked relative to the report.

Output
------
<results_dir>/debug/<game>/dataset/report.md
"""

import os
import re

import click
import pandas as pd

from utils import log_info, log_warn, log_error
from debug_scripts import markdown as md
from debug_scripts.paths import Paths
from debug_scripts.stats import gini

# Environment actions the low_level controller accepts; anything else in an `Action:` line
# is a target the agent could never execute.
VALID_ACTIONS = {"A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"}

PROMPT_BLOCKS = {
    "Recent actions": "history section (HistoryAwareExecutor)",
    "[ERROR]": "parse-failure feedback",
    "Tool result:": "tool output",
    "Available tool calls": "tool menu (absent when max_tool_calls=0)",
    "[HINT": "hint block (stripped by create_dataset)",
    "[STEP_INFO": "step-info block (stripped by create_dataset)",
}


def _parsed_action(output: str):
    """The `Action:` value from a training target, or None when there is no parseable line."""
    if not isinstance(output, str):
        return None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("action:"):
            value = stripped[len("action:"):].replace("[STOP]", "").strip()
            return value if value else None
    return None


def _audit(frame: pd.DataFrame) -> dict:
    actions = frame["output"].map(_parsed_action)
    unparseable = int(actions.isna().sum())
    normalised = actions.dropna().str.strip().str.upper()
    invalid = normalised[~normalised.isin(VALID_ACTIONS)]
    return {
        "counts": normalised.value_counts(),
        "unparseable": unparseable,
        "invalid": invalid.value_counts(),
        "n_invalid": int(len(invalid)),
    }


@click.command(name="dataset")
@click.option("--model_name", required=True, help="Full VLM name (e.g. google/gemma-4-31b-it)")
@click.option("--extra", default="zeroshot_with_curiosity", show_default=True,
              help="Which proposal variant's dataset to report.")
@click.option("--n_samples", default=50, show_default=True,
              help="Random rows to show, split across train and validation.")
@click.option("--seed", default=0, show_default=True, help="Seed for the sample.")
@click.pass_obj
def debug_dataset(obj, model_name, extra, n_samples, seed):
    """Show random train/validation rows verbatim, with an automatic target audit."""
    paths = Paths(
        parameters=obj["parameters"], game=obj["game"], run_name=obj["run_name"],
        executor=obj["executor"], model_name=model_name, output_dir=obj["output_dir"],
    )
    report_dir = paths.debug_dir("dataset")

    train_path = paths.require(paths.train_csv(extra), "dataset")
    val_path = paths.require(paths.validation_csv(extra), "dataset")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    log_info(f"[dataset] {len(train)} train / {len(val)} validation rows")

    if n_samples > len(train) + len(val):
        log_error(
            f"--n_samples {n_samples} exceeds the {len(train) + len(val)} rows available.",
            paths.parameters,
        )

    audit = _audit(train)
    total = len(train)
    rows_per_task = train["task_string"].value_counts()

    action_table = pd.DataFrame({
        "action": audit["counts"].index,
        "rows": audit["counts"].values,
        "share_%": audit["counts"].values / max(total, 1) * 100,
        "valid": [a in VALID_ACTIONS for a in audit["counts"].index],
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
                    parts.append(md.img("frame", image_path, report_dir))
                else:
                    missing_images += 1
                    parts.append(md.para(f"_(image missing: `{image_path}`)_"))
        action = _parsed_action(row["output"])
        flag = ""
        if action is None:
            flag = "  ⚠ **no parseable `Action:` line**"
        elif action.strip().upper() not in VALID_ACTIONS:
            flag = f"  ⚠ **`{action}` is not a valid environment action**"
        parts.append(md.para(f"**input**{flag}"))
        parts.append(md.code(row["input"]))
        parts.append(md.para("**output**"))
        parts.append(md.code(row["output"]))
        sample_blocks.append("\n".join(parts))

    if missing_images:
        log_warn(f"[dataset] {missing_images} referenced images are missing on disk")

    report_blocks = [
        md.h1(f"SFT dataset — {paths.game} / {paths.model_save_name} / extra={extra}"),
        md.para(f"Source: `{paths.practice_dir(extra)}`"),
        md.h2("Size"),
        md.bullets([
            f"train rows: **{len(train)}** over **{train['task_string'].nunique()}** task strings",
            f"validation rows: **{len(val)}** over **{val['task_string'].nunique()}** task strings",
            f"unique training targets (`output`): **{train['output'].nunique()}**",
            f"rows-per-task Gini: **{gini(rows_per_task.values):.3f}** "
            f"(top 10 tasks hold {rows_per_task.head(10).sum() / max(total, 1) * 100:.1f}% of rows)",
        ]),
        md.h2("Target audit (train)"),
        md.bullets([
            f"rows with **no parseable `Action:` line**: **{audit['unparseable']}** "
            f"({audit['unparseable'] / max(total, 1) * 100:.2f}%)",
            f"rows whose action is **not a valid environment action**: **{audit['n_invalid']}** "
            f"({audit['n_invalid'] / max(total, 1) * 100:.2f}%)",
        ]),
        md.para(
            "Both categories are targets the agent is trained to emit but the environment can "
            "never execute; at benchmark time they surface as `n_invalid`."
        ),
        md.table(action_table),
        md.details("invalid action strings",
                   md.table(pd.DataFrame({
                       "action": audit["invalid"].index,
                       "rows": audit["invalid"].values,
                   }))),
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

    report_path = md.write_report(os.path.join(report_dir, "report.md"), report_blocks)
    log_info(f"[dataset] wrote {report_path}")
    print(report_path)
