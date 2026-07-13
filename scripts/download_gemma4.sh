#!/bin/bash
# Download google/gemma-4-31B weights to HF cache.
# Run once on a CPU node (no GPU needed for download):
#   qsub -q all.q -l h_vmem=16G,h_rt=10800 -j y -cwd scripts/download_gemma4.sh
#
# ~62GB bfloat16 weights — expect 30-60 min depending on network.

cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate

echo "=== Downloading google/gemma-4-31B ==="
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'google/gemma-4-31B',
    cache_dir='/user/af3698/.cache/huggingface/hub',
    ignore_patterns=['*.pt', 'flax_model*', 'tf_model*'],
)
print('gemma-4-31B done')
"

echo "=== Done. Cache at /user/af3698/.cache/huggingface/hub ==="
