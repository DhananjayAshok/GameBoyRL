#!/bin/bash
#$ -q gpu.q@researchgpu05.grid.gsb
# Harness pipeline for Pokemon Prism (low-level + semantic action set)
# Uses PrismHarnessExecutor (prism_harness) with the propose_and_attempt pipeline.
#
# Submit: qsub -q gpu.q@researchgpu05.grid.gsb -l gpu=1,h_vmem=80G,h_rt=86400 -j y -cwd scripts/run_harness_prism.sh

cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY must be set in the environment}"

echo "=== GPU state at job start ==="
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader

# Try each GPU (skipping 0, which has persistent CUDA corruption) until vLLM
# successfully loads.  A pre-check then retry-on-crash handles the race where
# another job grabs the GPU between our check and the model-load phase.
VLLM_PID=""
VLLM_STARTED=0

for GPU_IDX in 1 2 3 4 5 6 7; do
  FREE=$(CUDA_VISIBLE_DEVICES=$GPU_IDX python -c \
    "import torch; free,_=torch.cuda.mem_get_info(0); print(int(free//1024//1024))" 2>/dev/null)
  if [ -z "$FREE" ] || [ "$FREE" -lt 20000 ]; then
    echo "GPU $GPU_IDX: ${FREE}MiB free — skipping"
    continue
  fi

  export CUDA_VISIBLE_DEVICES=$GPU_IDX
  echo "Trying GPU $GPU_IDX (${FREE}MiB free)..."

  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --quantization awq \
    --port 8000 \
    --max-model-len 8192 \
    --dtype float16 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --num-gpu-blocks-override 4000 &
  VLLM_PID=$!

  for i in $(seq 1 90); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
      echo "Server ready on GPU $GPU_IDX after ${i}x10s"
      VLLM_STARTED=1
      break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
      echo "vLLM crashed on GPU $GPU_IDX at iteration $i — trying next GPU"
      wait $VLLM_PID 2>/dev/null
      VLLM_PID=""
      break
    fi
    echo "  waiting... ($i/90) GPU=$GPU_IDX"
    sleep 10
  done

  [ $VLLM_STARTED -eq 1 ] && break

  # Kill in case wait loop timed out without crash
  kill $VLLM_PID 2>/dev/null
  wait $VLLM_PID 2>/dev/null
  VLLM_PID=""
done

if [ $VLLM_STARTED -eq 0 ]; then
  echo "ERROR: vLLM failed to start on any GPU (1-7)"
  exit 1
fi

echo "=== Running harness propose_and_attempt pipeline ==="
bash scripts/pipeline/propose_and_attempt.sh \
  --game pokemon_prism \
  --model_name Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --vlm_kind vllm \
  --init_states starter \
  --executor prism_harness \
  --controller_variant state_wise \
  --max_steps 100 \
  --max_attempts 3 \
  --do_guidance_and_practice false \
  --verbose true \
  --overwrite false \
  2>&1

PIPELINE_EXIT=$?
echo "PIPELINE EXIT CODE: $PIPELINE_EXIT"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
exit $PIPELINE_EXIT
