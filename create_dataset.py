"""
Create a VLA training dataset from successful practice trajectories, with a
leakage-free train/validation split and paraphrase augmentation on train only.

Input
-----
A practice output directory produced by vlm_scripts/practice_tasks.py
(--practice_path). Contains results.csv and {group_idx}_{attempt}.pkl files
(each pkl is a List[VLMCallRecord] for that episode).

It also expects the artifacts produced by vlm_scripts/clean_practice.py in the
same directory (run clean_practice first):
  paraphrases.json     — {task_string: [paraphrase, ...]} used to augment train.
  clean_decisions.csv  — group_idx, attempt, call_idx, accept (bool); rows with
    accept == False are dropped from the dataset.
Both are required: a missing paraphrases.json or clean_decisions.csv is an error
(run clean_practice on the practice dir first).

Output
------
Written into --practice_path:
  train_dataset.csv      — training rows, augmented with all but the last
    paraphrase of each task.
  validation_dataset.csv — held-out rows, each phrased with the task's RESERVED
    last paraphrase (never used in train), for leakage-free model selection in
    train_vlm.sh. Tasks with no paraphrase fall back to the canonical string.
  images/ — JPEG files named {group_idx}_{attempt}_{call_idx}_{img_idx}.jpg

Row fields: task_string, input, output, image, score.

Paraphrases
-----------
For each task, paraphrases.json holds up to k paraphrases. The LAST one is
reserved for validation only: train rows are augmented with paraphrases[:-1],
and validation rows use paraphrases[-1] as their task string. So neither the
held-out episodes nor the held-out phrasing are ever seen during training.

Split
-----
Episodes are split per task: for each task_string, its episodes (group_idx,
attempt) are shuffled (seeded) and val_frac of them go to validation, the rest
to train. The split is by *episode*, so all calls of an episode stay on one side
and no episode (hence no shared output/image) straddles the split. If a task has
too few episodes to yield at least one validation episode
(floor(n_episodes * val_frac) < 1), all of its episodes go to train.

Usage
-----
python create_dataset.py --practice_path <dir> [--overwrite] [--val_frac 0.2]
"""

import json
import os
import pickle
import re
from collections import defaultdict

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import log_error, log_info


HINT_RE = re.compile(r'\n?\[HINT_START\].*?\[HINT_END\]', re.DOTALL)
STEP_INFO_RE = re.compile(r'\n?\[STEP_INFO\].*?\[STEP_INFO_END\]', re.DOTALL)


def _strip_blocks(text: str) -> str:
    return STEP_INFO_RE.sub('', HINT_RE.sub('', text))


def _episode_cutoff(safe_success_point, n_calls: int, safety_margin: int) -> int:
    """Return the number of leading VLM calls to keep for an episode.

    ``safe_success_point`` is already a vlm_call_log slice index (see
    execution/supervisor.py), not a frame number, so we slice directly.
    ``safety_margin`` keeps a few extra calls as insurance. Returns ``n_calls``
    (keep everything) when safe_success_point is N/A.
    """
    if safe_success_point is None or pd.isna(safe_success_point):
        return n_calls
    return min(n_calls, int(safe_success_point) + safety_margin)


def select_successful(df: pd.DataFrame, score_threshold: float = 6.0) -> pd.DataFrame:
    """Episodes usable for the dataset, robust to both practice scoring modes.

    practice_tasks writes results.csv in one of two shapes:
      * binary mode — ``success`` in {True, False}, ``score`` is NaN.
      * score mode  — ``success`` is NaN, ``score`` in 1..10.
    An episode is usable if ``success`` is True (binary) OR, when ``success`` is
    absent/NaN (score mode), ``score >= score_threshold``. The threshold mirrors
    practice_tasks' ``failed = score < 6`` convention.

    This is the single source of truth for episode selection: clean_practice
    imports it so the two stay in lockstep. It MUST be called with the same
    ``score_threshold`` in both, or clean would judge a different episode set than
    create_dataset emits.
    """
    success = df['success'] if 'success' in df.columns else pd.Series(np.nan, index=df.index)
    score = df['score'] if 'score' in df.columns else pd.Series(np.nan, index=df.index)
    binary_ok = success == True
    score_ok = success.isna() & (score >= score_threshold)
    return df[binary_ok | score_ok]


def _save_image(img_array: np.ndarray, path: str) -> None:
    arr = img_array
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 255)).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode='L').convert('RGB')
    elif arr.shape[-1] == 1:
        img = Image.fromarray(arr[..., 0]).convert('RGB')
    else:
        img = Image.fromarray(arr)
    img.save(path, 'JPEG')


def _load_decisions(practice_path: str) -> dict:
    """Load clean_decisions.csv into {(group_idx, attempt, call_idx): accept_bool}."""
    path = os.path.join(practice_path, 'clean_decisions.csv')
    if not os.path.exists(path):
        log_error(
            f"clean_decisions.csv not found at {path}. Run clean_practice on this "
            "practice dir before create_dataset."
        )
    df = pd.read_csv(path)
    return {
        (str(r['group_idx']), int(r['attempt']), int(r['call_idx'])): bool(r['accept'])
        for _, r in df.iterrows()
    }


def _load_paraphrases(practice_path: str) -> dict:
    """Load paraphrases.json into {task_string: [paraphrase, ...]}."""
    path = os.path.join(practice_path, 'paraphrases.json')
    if not os.path.exists(path):
        log_error(
            f"paraphrases.json not found at {path}. Run clean_practice on this "
            "practice dir before create_dataset."
        )
    with open(path, 'r') as f:
        return json.load(f)


def _split_episodes_by_task(episodes_by_task: dict, val_frac: float, seed: int) -> set:
    """Return the set of episode ids assigned to validation.

    For each task, shuffle its episodes and hold out floor(n * val_frac); if that
    is < 1 the task contributes nothing to validation (all its episodes train).
    """
    rng = np.random.default_rng(seed)
    val_episodes = set()
    for task, episodes in episodes_by_task.items():
        episodes = sorted(episodes)
        n_val = int(len(episodes) * val_frac)
        if n_val < 1:
            continue
        perm = rng.permutation(len(episodes))
        for i in perm[:n_val]:
            val_episodes.add(episodes[i])
    return val_episodes


@click.command()
@click.option('--practice_path', required=True, type=str,
              help='Path to the practice output directory (contains results.csv and *.pkl).')
@click.option('--overwrite', is_flag=True, default=False,
              help='Overwrite existing output files.')
@click.option('--safety_margin', default=2, show_default=True, type=int,
              help='Extra VLM calls kept past the safe-success cutoff (see _episode_cutoff). '
                   'Should match clean_practice --safety_margin.')
@click.option('--val_frac', default=0.2, show_default=True, type=float,
              help='Fraction of each task\'s episodes held out for validation.')
@click.option('--seed', default=0, show_default=True, type=int,
              help='Seed for the per-task episode shuffle.')
@click.option('--score_threshold', default=6.0, show_default=True, type=float,
              help='Min score for a score-mode episode to count as successful '
                   '(ignored in binary mode). Should match clean_practice --score_threshold.')
def create_dataset(practice_path, overwrite, safety_margin, val_frac, seed, score_threshold):
    """Build train/validation VLA datasets from successful practice episodes."""
    results_csv = os.path.join(practice_path, 'results.csv')
    if not os.path.exists(results_csv):
        log_error(f"results.csv not found in {practice_path}")

    images_dir = os.path.join(practice_path, 'images')
    train_path = os.path.join(practice_path, 'train_dataset.csv')
    val_path = os.path.join(practice_path, 'validation_dataset.csv')

    if (os.path.exists(train_path) or os.path.exists(val_path)) and not overwrite:
        log_info(f"Dataset already exists in {practice_path}. Use --overwrite to regenerate.")
        return

    df = pd.read_csv(results_csv)
    successful = select_successful(df, score_threshold)
    log_info(f"Total episodes: {len(df)} | Successful: {len(successful)}")

    decisions = _load_decisions(practice_path)
    paraphrases = _load_paraphrases(practice_path)

    os.makedirs(images_dir, exist_ok=True)

    # Build one row per kept call, tagging each with its episode id and task so we
    # can split by episode afterwards.
    rows = []
    episodes_by_task = defaultdict(set)
    missing_pkls = 0
    n_rejected = 0
    for _, row in tqdm(successful.iterrows(), total=len(successful), desc='episodes'):
        # group_idx is a string id like "10_0"; int() would mangle it via underscore
        # digit-separator parsing (int("10_0") == 100). Keep it as a raw string.
        group_idx = str(row['group_idx'])
        attempt = int(row['attempt'])
        episode_id = (group_idx, attempt)
        pkl_path = os.path.join(practice_path, f'{group_idx}_{attempt}.pkl')

        if not os.path.exists(pkl_path):
            missing_pkls += 1
            continue

        with open(pkl_path, 'rb') as f:
            vlm_call_log = pickle.load(f)

        cutoff = _episode_cutoff(row.get('safe_success_point'), len(vlm_call_log), safety_margin)

        for call_idx, record in enumerate(vlm_call_log[:cutoff]):
            if decisions.get((group_idx, attempt, call_idx)) is False:
                n_rejected += 1
                continue

            image_paths = []
            for img_idx, img_array in enumerate(record.images):
                img_fname = f'{group_idx}_{attempt}_{call_idx}_{img_idx}.jpg'
                img_path = os.path.join(images_dir, img_fname)
                _save_image(img_array, img_path)
                image_paths.append(img_path)

            episodes_by_task[row['task_string']].add(episode_id)
            rows.append({
                'episode_id': episode_id,
                'task_string': row['task_string'],
                'input': _strip_blocks(record.prompt),
                'output': record.response,
                'image': ', '.join(image_paths),
                'score': row['score'],
            })

    val_episodes = _split_episodes_by_task(episodes_by_task, val_frac, seed)

    fields = ['task_string', 'input', 'output', 'image', 'score']
    train_rows, val_rows = [], []
    for row in rows:
        clean_row = {k: row[k] for k in fields}
        task = row['task_string']
        task_paraphrases = paraphrases.get(task, [])
        # Reserve the last paraphrase for validation only: train never uses it,
        # and validation rows are phrased with it (an unseen phrasing). The rest
        # augment train.
        test_paraphrase = task_paraphrases[-1] if task_paraphrases else None
        train_paraphrases = task_paraphrases[:-1]

        if row['episode_id'] in val_episodes:
            if test_paraphrase is not None:
                val_row = dict(clean_row)
                val_row['task_string'] = test_paraphrase
                val_row['input'] = row['input'].replace(task, test_paraphrase)
                val_rows.append(val_row)
            else:
                # No paraphrase available for this task — fall back to canonical.
                val_rows.append(clean_row)
        else:
            train_rows.append(clean_row)
            # Augment train only, never with the reserved test paraphrase.
            for paraphrase in train_paraphrases:
                aug = dict(clean_row)
                aug['task_string'] = paraphrase
                aug['input'] = row['input'].replace(task, paraphrase)
                train_rows.append(aug)

    pd.DataFrame(train_rows, columns=fields).to_csv(train_path, index=False)
    pd.DataFrame(val_rows, columns=fields).to_csv(val_path, index=False)

    if missing_pkls:
        log_info(f"Warning: {missing_pkls} pkl files not found (skipped).")
    log_info(f"Dropped {n_rejected} rejected calls.")
    log_info(f"Episodes: {len(val_episodes)} validation / "
             f"{sum(len(e) for e in episodes_by_task.values()) - len(val_episodes)} train.")
    log_info(f"Saved {len(train_rows)} train rows -> {train_path}")
    log_info(f"Saved {len(val_rows)} validation rows -> {val_path}")
    log_info(f"Images -> {images_dir}/")


if __name__ == '__main__':
    create_dataset()
