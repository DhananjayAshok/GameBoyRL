"""
Create a VLA training dataset from successful practice trajectories.

Input
-----
A practice output directory produced by vlm_scripts/practice_tasks.py.
Contains results.csv and {group_idx}_{attempt}.pkl files (each pkl is a
List[VLMCallRecord] for that episode).

Output
------
Directory: parameters["storage_dir"]/datasets/<relpath_from_storage_dir>/
  dataset.csv — one row per VLM call across all successful episodes.
    Fields:
      input   — prompt text with hint block stripped
      output  — raw VLM response text
      images  — JSON-encoded list of absolute paths to saved JPEG frames

  images/ — JPEG files named {group_idx}_{attempt}_{call_idx}_{img_idx}.jpg

Usage
-----
python create_dataset.py --practice_path <path_to_practice_dir> [--overwrite]
"""

import os
import pickle
import re

import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import load_parameters


HINT_RE = re.compile(r'\n?\[HINT_START\].*?\[HINT_END\]', re.DOTALL)


def _strip_hint(text: str) -> str:
    return HINT_RE.sub('', text)


def _derive_output_dir(practice_path: str, storage_dir: str) -> str:
    abs_practice = os.path.realpath(practice_path)
    abs_storage = os.path.realpath(storage_dir)
    rel = os.path.relpath(abs_practice, abs_storage)
    if not rel.startswith('..'):
        return os.path.join(storage_dir, 'datasets', rel)
    return os.path.join(storage_dir, 'datasets', os.path.basename(abs_practice))


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


@click.command()
@click.option('--practice_path', required=True, type=str,
              help='Path to the practice output directory (contains results.csv and *.pkl).')
@click.option('--overwrite', is_flag=True, default=False,
              help='Overwrite existing dataset.')
def create_dataset(practice_path, overwrite):
    """Build a VLA training dataset from successful practice episode pkl files."""
    parameters = load_parameters()
    storage_dir = parameters['storage_dir']

    results_csv = os.path.join(practice_path, 'results.csv')
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"results.csv not found in {practice_path}")

    df = pd.read_csv(results_csv)
    successful = df[df['success'] == True]
    print(f"Total episodes: {len(df)} | Successful: {len(successful)}")

    out_dir = _derive_output_dir(practice_path, storage_dir)
    images_dir = os.path.join(out_dir, 'images')
    csv_path = os.path.join(out_dir, 'dataset.csv')

    if os.path.exists(csv_path) and not overwrite:
        print(f"Dataset already exists at {csv_path}. Use --overwrite to regenerate.")
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

        for call_idx, record in enumerate(vlm_call_log):
            image_paths = []
            for img_idx, img_array in enumerate(record.images):
                img_fname = f'{group_idx}_{attempt}_{call_idx}_{img_idx}.jpg'
                img_path = os.path.join(images_dir, img_fname)
                _save_image(img_array, img_path)
                image_paths.append(img_path)

            rows.append({
                'input': _strip_hint(record.prompt),
                'output': record.response,
                'image': ', '.join(image_paths),
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    if missing_pkls:
        print(f"Warning: {missing_pkls} pkl files not found (skipped).")
    print(f"Saved {len(rows)} records -> {csv_path}")
    print(f"Images -> {images_dir}/")


if __name__ == '__main__':
    create_dataset()
