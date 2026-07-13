#!/bin/bash
# Pre-download Qwen2-VL-7B and Qwen2.5-VL-32B model weights to HF cache
# Run this ONCE on a GPU/CPU node so subsequent jobs don't need network access
cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate

echo "=== Downloading Qwen2-VL-7B-Instruct ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2-VL-7B-Instruct',
    cache_dir='/user/af3698/.cache/huggingface/hub',
    ignore_patterns=['*.pt', 'flax_model*', 'tf_model*'],
)
print('Qwen2-VL-7B done')
"

echo "=== Downloading Qwen2.5-VL-32B-Instruct AWQ (4-bit, ~18GB) ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-VL-32B-Instruct-AWQ',
    cache_dir='/user/af3698/.cache/huggingface/hub',
    ignore_patterns=['*.pt', 'flax_model*', 'tf_model*'],
)
print('Qwen2.5-VL-32B-AWQ done')
"

echo "=== Done. Cache at /user/af3698/.cache/huggingface/hub ==="
