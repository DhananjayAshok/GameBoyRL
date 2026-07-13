#!/bin/bash
#$ -q gpu.q@researchgpu05.grid.gsb
# propose_and_attempt pipeline for pokemon_prism (32B AWQ vLLM)
# Phase 1: propose zero-shot tasks from starter init_state
# Phase 2: attempt each task with prism_badge executor (state_wise controller)
#
# Submit: qsub -q gpu.q@researchgpu05.grid.gsb -l gpu=1,h_vmem=80G,h_rt=86400 -j y -cwd scripts/run_propose_and_attempt_prism.sh

cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY must be set in the environment}"

# Pin to GPU 1 — GPU 0 has persistent CUDA context corruption on researchgpu05
export CUDA_VISIBLE_DEVICES=1

echo "=== GPU state at job start ==="
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

FREE_MIB=$(python -c "import torch; free,_ = torch.cuda.mem_get_info(0); print(int(free//1024//1024))" 2>/dev/null)
if [ -z "$FREE_MIB" ] || [ "$FREE_MIB" -lt 20000 ]; then
  echo "ERROR: Insufficient GPU memory on CUDA:0 (physical GPU ${CUDA_VISIBLE_DEVICES}). Free=${FREE_MIB}MiB, need >=20000MiB."
  exit 2
fi
echo "GPU memory check passed: ${FREE_MIB}MiB free"

echo "=== Starting vLLM server with Qwen2.5-VL-32B-Instruct-AWQ ==="
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

echo "=== Waiting for vLLM server ready (15 min max) ==="
for i in $(seq 1 90); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Server ready after ${i}x10s"
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM process died at iteration $i"
    exit 1
  fi
  echo "  waiting... ($i/90)"
  sleep 10
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
  echo "ERROR: vLLM server never became ready"
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

echo "=== Running propose_and_attempt pipeline ==="
bash scripts/pipeline/propose_and_attempt.sh \
  --game pokemon_prism \
  --model_name Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --vlm_kind vllm \
  --init_states starter \
  --executor prism_badge \
  --controller_variant state_wise \
  --max_steps 75 \
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
