"""
Create a VLA training dataset from successful practice trajectories.

Input
-----
A practice output directory produced by vlm_scripts/practice_tasks.py.
Contains results.csv and {group_idx}_{attempt}.pkl files (each pkl is a
List[VLMCallRecord] for that episode).

Output
------
All outputs are written directly into the practice directory (--practice_path).
  dataset.csv — one row per VLM call across all successful episodes, truncated
    at each episode's safe-success cutoff (+ --safety_margin) so post-success
    overshoot is excluded. The cutoff comes from the results.csv
    ``safe_success_point`` column, which (despite its name) holds a vlm_call_log
    index, not a frame number — see _episode_cutoff and execution/supervisor.py.
    Fields:
      task_string — the practice task string for this episode
      input       — prompt text with hint block and STEP_INFO stripped
      output      — raw VLM response text
      image       — comma-joined list of absolute paths to saved JPEG frames
      score       — episode score from results.csv

  augmented_dataset.csv — dataset.csv plus paraphrase-augmented copies. For each
    unique task_string a VLM generates several paraphrases; for every original
    row, k=3 of them are added as new rows with the task substring substituted
    inside the ``input`` prompt and ``task_string`` set to the paraphrase. All
    other fields (output, image, score) are kept identical.

  images/ — JPEG files named {group_idx}_{attempt}_{call_idx}_{img_idx}.jpg

Usage
-----
python create_dataset.py --practice_path <dir> [--overwrite] create_dataset
python create_dataset.py --practice_path <dir> [--overwrite] augment_dataset \\
    --model_name <name> --vlm_kind <kind>
"""

import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import VLM, HuggingFaceModel, log_info


HINT_RE = re.compile(r'\n?\[HINT_START\].*?\[HINT_END\]', re.DOTALL)
STEP_INFO_RE = re.compile(r'\n?\[STEP_INFO\].*?\[STEP_INFO_END\]', re.DOTALL)

PARAPHRASE_PROMPT = """You are given a canonical task string:
"[CORE_TASK]"

Generate at least [N] diverse, valid paraphrases of this task. Vary the wording,
phrasing style, and structure but preserve the exact core meaning and level of specificity.
Use imperative tone throughout.

Respond in exactly this format (one paraphrase per line):
- <paraphrase 1>
- <paraphrase 2>
...
[STOP]"""


def _strip_blocks(text: str) -> str:
    return STEP_INFO_RE.sub('', HINT_RE.sub('', text))


def _parse_bullet_list(text: str) -> list[str]:
    """Return all '- ...' bullet lines before [STOP]. Preserves original casing."""
    results = []
    for line in text.splitlines():
        if "[stop]" in line.lower():
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            results.append(stripped[2:].strip())
    return results


def _episode_cutoff(safe_success_point, n_calls: int, safety_margin: int) -> int:
    """Return the number of leading VLM calls to keep for an episode.

    Despite its name, ``safe_success_point`` here is NOT a frame number — it is
    already a vlm_call_log slice index. The checker reports a frame number, but
    execution/supervisor.py converts it to an exact call-log index (via the
    lockstep frame→call walk, while it still has the steps list) and stores that
    index under the ``safe_success_point`` key. So we can slice directly; no
    frame-vs-call approximation happens here. Everything past the cutoff is
    overshoot we don't want in the training set.

    ``safety_margin`` keeps a few extra calls past the cutoff as insurance.
    Returns ``n_calls`` (keep everything) when safe_success_point is N/A — either
    the checker couldn't pin down a completion frame, or the column is absent in
    an older results.csv produced before this conversion was added.
    """
    if safe_success_point is None or pd.isna(safe_success_point):
        return n_calls
    return min(n_calls, int(safe_success_point) + safety_margin)


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


def _paraphrase_task(task: str, vlm: VLM, max_new_tokens: int, k: int) -> list[str]:
    """Ask the VLM for paraphrases of a single task string. Returns up to k of them."""
    prompt = (
        PARAPHRASE_PROMPT
        .replace("[CORE_TASK]", task)
        .replace("[N]", str(k))
    )
    output = vlm.infer(texts=prompt, max_new_tokens=max_new_tokens)
    paraphrases = [p for p in _parse_bullet_list(output) if p and p != task]
    return paraphrases[:k]


@click.group()
@click.option('--practice_path', required=True, type=str,
              help='Path to the practice output directory (contains results.csv and *.pkl).')
@click.option('--overwrite', is_flag=True, default=False,
              help='Overwrite existing output files.')
@click.pass_context
def cli(ctx, practice_path, overwrite):
    """Build and augment VLA training datasets from practice trajectories."""
    ctx.obj = dict(
        practice_path=practice_path,
        overwrite=overwrite,
    )


@cli.command(name='create_dataset')
@click.option('--safety_margin', default=2, show_default=True, type=int,
              help='Extra VLM calls kept past the safe-success cutoff as insurance '
                   '(see _episode_cutoff).')
@click.pass_obj
def create_dataset(obj, safety_margin):
    """Build a VLA training dataset from successful practice episode pkl files."""
    practice_path = obj['practice_path']
    overwrite = obj['overwrite']

    results_csv = os.path.join(practice_path, 'results.csv')
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"results.csv not found in {practice_path}")

    df = pd.read_csv(results_csv)
    successful = df[df['success'] == True]
    log_info(f"Total episodes: {len(df)} | Successful: {len(successful)}")

    out_dir = practice_path
    images_dir = os.path.join(out_dir, 'images')
    csv_path = os.path.join(out_dir, 'dataset.csv')

    if os.path.exists(csv_path) and not overwrite:
        log_info(f"Dataset already exists at {csv_path}. Use --overwrite to regenerate.")
        return

    os.makedirs(images_dir, exist_ok=True)

    rows = []
    missing_pkls = 0
    for _, row in tqdm(successful.iterrows(), total=len(successful), desc='episodes'):
        group_idx = str(int(row['group_idx']))
        attempt = int(row['attempt'])
        pkl_path = os.path.join(practice_path, f'{group_idx}_{attempt}.pkl')

        if not os.path.exists(pkl_path):
            missing_pkls += 1
            continue

        with open(pkl_path, 'rb') as f:
            vlm_call_log = pickle.load(f)

        cutoff = _episode_cutoff(row.get('safe_success_point'), len(vlm_call_log), safety_margin)

        for call_idx, record in enumerate(vlm_call_log[:cutoff]):
            image_paths = []
            for img_idx, img_array in enumerate(record.images):
                img_fname = f'{group_idx}_{attempt}_{call_idx}_{img_idx}.jpg'
                img_path = os.path.join(images_dir, img_fname)
                _save_image(img_array, img_path)
                image_paths.append(img_path)

            rows.append({
                "task_string": row['task_string'],
                'input': _strip_blocks(record.prompt),
                'output': record.response,
                'image': ', '.join(image_paths),
                "score": row['score'],
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    if missing_pkls:
        log_info(f"Warning: {missing_pkls} pkl files not found (skipped).")
    log_info(f"Saved {len(rows)} records -> {csv_path}")
    log_info(f"Images -> {images_dir}/")


@cli.command(name='augment_dataset')
@click.option('--model_name', required=True, help='VLM model name (e.g. gpt-4o).')
@click.option('--vlm_kind', required=True,
              type=click.Choice(['openai', 'anthropic', 'openrouter', 'huggingface', 'vllm']),
              help='VLM backend kind.')
@click.option('--max_new_tokens', default=1000, show_default=True, type=int,
              help='Max tokens for each VLM call.')
@click.option('--k', default=3, show_default=True, type=int,
              help='Number of paraphrase copies to add per original row.')
@click.option('--max_concurrency', default=16, show_default=True, type=int,
              help='Max concurrent paraphrase calls. Forced to 1 for a huggingface vlm_kind.')
@click.pass_obj
def augment_dataset(obj, model_name, vlm_kind, max_new_tokens, k, max_concurrency):
    """Add k paraphrase-augmented copies of each row to augmented_dataset.csv."""
    overwrite = obj['overwrite']

    out_dir = obj['practice_path']
    dataset_path = os.path.join(out_dir, 'dataset.csv')
    augmented_path = os.path.join(out_dir, 'augmented_dataset.csv')

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"dataset.csv not found at {dataset_path}. Run create_dataset first."
        )
    if os.path.exists(augmented_path) and not overwrite:
        log_info(f"Augmented dataset already exists at {augmented_path}. Use --overwrite to regenerate.")
        return

    df = pd.read_csv(dataset_path)
    unique_tasks = [t for t in df['task_string'].dropna().unique()]
    log_info(f"Loaded {len(df)} rows | {len(unique_tasks)} unique tasks")

    vlm = VLM(model_name, vlm_kind)
    # Paraphrase calls are independent across tasks, so submit them all to one
    # shared pool. HuggingFaceModel isn't safe for concurrent generate() calls,
    # so fall back to a single worker (matching infer_tasks).
    effective_workers = 1 if isinstance(vlm._vlm, HuggingFaceModel) else max_concurrency

    task_to_paraphrases = {}
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            task: executor.submit(_paraphrase_task, task, vlm, max_new_tokens, k)
            for task in unique_tasks
        }
        for task, future in tqdm(futures.items(), desc='paraphrasing tasks'):
            paraphrases = future.result()
            if len(paraphrases) < k:
                log_info(f"Warning: only {len(paraphrases)} paraphrase(s) for task {task!r}.")
            task_to_paraphrases[task] = paraphrases
    augmented_rows = []
    for _, row in df.iterrows():
        augmented_rows.append(row.to_dict())
        task = row['task_string']
        for paraphrase in task_to_paraphrases.get(task, []):
            new_row = row.to_dict()
            new_row['task_string'] = paraphrase
            new_row['input'] = row['input'].replace(task, paraphrase)
            augmented_rows.append(new_row)

    pd.DataFrame(augmented_rows).to_csv(augmented_path, index=False)
    log_info(f"Saved {len(augmented_rows)} records ({len(df)} original) -> {augmented_path}")


if __name__ == '__main__':
    cli()
