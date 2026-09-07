#!/bin/bash
#$ -q gpu.q@researchgpu05.grid.gsb
# Exercises the accordion on hardware: ONE control run with more tasks than
# COMPRESS_THRESHOLD (10), so the attempt log actually folds. Every earlier run stopped at
# 8 tasks and never reached the fold, leaving compression offline-tested only.
#
# max_steps_per_task is cut to 40 (from 175) purely for wall-clock. The fold triggers on
# the ATTEMPT COUNT, not on steps, so a shorter step budget does not weaken the test — it
# just means each attempt achieves less, which is fine when the goal is not the point.
# No ablation arm: it says nothing about compression.
#
# Submit:
#   qsub -l gpu=1,h_vmem=80G,h_rt=86400 -j y -cwd scripts/run_strategist_compression.sh

cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate
export HF_HOME="/scratch/af3698/huggingface_cache"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-required-using-vllm}"

MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
GAME="pokemon_red"
GOAL="Obtain your first Pokemon"
EXECUTOR="single_actions"
MAX_TASKS=12      # > COMPRESS_THRESHOLD (10): the log folds once, at attempt 10
MAX_STEPS=40      # wall-clock only; does not affect when the fold fires

echo "=== GPU state at job start ==="
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader

# --- Start vLLM ---
VLLM_PID=""
VLLM_STARTED=0

for GPU_IDX in 0 1 2 3 4 5 6 7; do
  FREE=$(CUDA_VISIBLE_DEVICES=$GPU_IDX python -c \
    "import torch; free,_=torch.cuda.mem_get_info(0); print(int(free//1024//1024))" 2>/dev/null)
  if [ -z "$FREE" ] || [ "$FREE" -lt 20000 ]; then
    echo "GPU $GPU_IDX: ${FREE:-0}MiB free — skipping"
    continue
  fi

  export CUDA_VISIBLE_DEVICES=$GPU_IDX
  echo "Trying GPU $GPU_IDX (${FREE}MiB free)..."

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
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
      echo "vLLM ready on GPU $GPU_IDX after ${i}x10s"
      VLLM_STARTED=1
      break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
      echo "vLLM crashed on GPU $GPU_IDX — trying next"
      wait $VLLM_PID 2>/dev/null
      VLLM_PID=""
      break
    fi
    echo "  waiting... ($i/90)"
    sleep 10
  done

  [ $VLLM_STARTED -eq 1 ] && break
  kill $VLLM_PID 2>/dev/null
  wait $VLLM_PID 2>/dev/null
  VLLM_PID=""
done

if [ $VLLM_STARTED -eq 0 ]; then
  echo "ERROR: vLLM failed to start on any GPU"
  exit 1
fi

# --- The compression test: more tasks than the fold threshold ---
echo ""
echo "=== STRATEGIST COMPRESSION TEST (12 tasks, fold threshold 10) ==="
python run_strategist.py \
  --game "$GAME" \
  --goal "$GOAL" \
  --executor "$EXECUTOR" \
  --executor_vlm_model "$MODEL" \
  --executor_vlm_kind vllm \
  --max_tasks "$MAX_TASKS" \
  --max_steps_per_task "$MAX_STEPS" \
  --use_notebook true \
  --verify_goal true \
  --save_video true \
  --extra_name compressiontest \
  --verbose
RUN_EXIT=$?
echo "Run exit: $RUN_EXIT"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM stopped."

echo ""
echo "=== Records written ==="
ls -la storage/strategist/$GAME/ 2>/dev/null | tail -5

echo ""
echo "=== DONE ==="
exit $RUN_EXIT
