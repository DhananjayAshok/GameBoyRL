#!/bin/bash
#$ -q gpu.q@researchgpu05.grid.gsb
# Harness pipeline for Pokemon Prism using google/gemma-4-31B.
# Needs 2 GPUs for bfloat16 (31B × 2 bytes ≈ 62GB; 2×A40 = 96GB headroom).
#
# Submit:
#   qsub -q gpu.q@researchgpu05.grid.gsb -l gpu=2,h_vmem=100G,h_rt=86400 -j y -cwd scripts/run_harness_gemma4.sh

cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY must be set in the environment}"

echo "=== GPU state at job start ==="
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader

# Need two free GPUs for tensor-parallel-size=2.
# Find the first two GPUs (skipping 0) that each have >= 40 GiB free.
FREE_GPUS=()
for GPU_IDX in 1 2 3 4 5 6 7; do
  FREE=$(CUDA_VISIBLE_DEVICES=$GPU_IDX python -c \
    "import torch; free,_=torch.cuda.mem_get_info(0); print(int(free//1024//1024))" 2>/dev/null)
  if [ -n "$FREE" ] && [ "$FREE" -ge 40000 ]; then
    FREE_GPUS+=($GPU_IDX)
    echo "GPU $GPU_IDX: ${FREE}MiB free — candidate"
  else
    echo "GPU $GPU_IDX: ${FREE}MiB free — skipping"
  fi
  [ ${#FREE_GPUS[@]} -ge 2 ] && break
done

if [ ${#FREE_GPUS[@]} -lt 2 ]; then
  echo "ERROR: need 2 GPUs with >=40GiB free, only found ${#FREE_GPUS[@]}"
  exit 1
fi

GPU_A=${FREE_GPUS[0]}
GPU_B=${FREE_GPUS[1]}
export CUDA_VISIBLE_DEVICES="${GPU_A},${GPU_B}"
echo "Using GPUs ${GPU_A} and ${GPU_B} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"

VLLM_PID=""
VLLM_STARTED=0

python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code &
VLLM_PID=$!

echo "vLLM starting (PID $VLLM_PID) on GPUs ${GPU_A},${GPU_B}..."

for i in $(seq 1 120); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Server ready after ${i}x10s"
    VLLM_STARTED=1
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM crashed at iteration $i"
    wait $VLLM_PID 2>/dev/null
    break
  fi
  echo "  waiting... ($i/120)"
  sleep 10
done

if [ $VLLM_STARTED -eq 0 ]; then
  echo "ERROR: vLLM failed to start"
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

echo "=== Running harness propose_and_attempt pipeline ==="
bash scripts/pipeline/propose_and_attempt.sh \
  --game pokemon_prism \
  --model_name google/gemma-4-31B \
  --vlm_kind vllm \
  --init_states starter \
  --executor prism_harness \
  --controller_variant state_wise \
  --max_steps 100 \
  --max_attempts 3 \
  --do_guidance_and_practice false \
  --verbose true \
  --overwrite true \
  2>&1

PIPELINE_EXIT=$?
echo "PIPELINE EXIT CODE: $PIPELINE_EXIT"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
exit $PIPELINE_EXIT
