#!/bin/bash
# Sets up flash attention and vLLM installs on the specific GPU nodes. Must be run on the GPU node itself. 

module load gcc/13.3.0 cuda/12.9.1
source setup/.venv/bin/activate
uv pip install --force-reinstall --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu129
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4
uv pip uninstall flash-attn vllm
uv pip install flash-attn --no-binary :all: --no-cache --no-build-isolation
cd vllm
uv pip install -e .
cd ..
echo "Done installing vLLM and flash attention."