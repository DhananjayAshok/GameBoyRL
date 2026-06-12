module load gcc/13.3.0 cuda/12.6.3
source setup/.venv/bin/activate
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4
uv pip uninstall flash-attn vllm
uv pip install flash-attn --no-binary :all: --no-cache --no-build-isolation
cd vllm
uv pip install -e .
cd ..
echo "Done installing vLLM and flash attention."