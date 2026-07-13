#!/bin/bash
#$ -q gpu.q@researchgpu05.grid.gsb
# Run vLLM server with Qwen2.5-VL-32B-AWQ + benchmark on pokemon_prism (prism_badge)
# Submit with: qsub -q gpu.q@researchgpu05.grid.gsb -l gpu=1,h_vmem=80G -j y -cwd scripts/run_vllm_benchmark_32b.sh
cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY must be set in the environment}"
# Pin to GPU 1 (physical) — GPU 0 on researchgpu05 has persistent CUDA context corruption.
export CUDA_VISIBLE_DEVICES=1

echo "=== GPU state at job start ==="
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Check free memory on the ACTUAL CUDA device via PyTorch
FREE_MIB=$(python -c "import torch; free,_ = torch.cuda.mem_get_info(0); print(int(free//1024//1024))" 2>/dev/null)
if [ -z "$FREE_MIB" ] || [ "$FREE_MIB" -lt 20000 ]; then
  echo "ERROR: Insufficient GPU memory on CUDA:0 (physical GPU ${CUDA_VISIBLE_DEVICES}). Free=${FREE_MIB}MiB, need >=20000MiB."
  exit 2
fi
echo "GPU memory check passed (PyTorch): ${FREE_MIB}MiB free on CUDA:0 = physical GPU ${CUDA_VISIBLE_DEVICES}"

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

echo "=== Waiting for server ready (15 min max) ==="
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

echo "=== Running benchmark: pokemon_prism, state_wise, prism_badge ==="
python run_benchmark.py \
  --game pokemon_prism \
  --executor prism_badge \
  --controller_variant state_wise \
  --executor_vlm_model Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --executor_vlm_kind vllm \
  --max_steps 300 \
  --save_video 1 \
  --regenerate \
  --task_text "Earn the Magma Badge from Gym Leader Tansy in Brimstone City" \
  --init_state starter \
  --state_tracker_class first_badge_test \
  2>&1

BENCH_EXIT=$?
echo "BENCHMARK EXIT CODE: $BENCH_EXIT"
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
exit $BENCH_EXIT
