#!/bin/bash
#$ -q gpu.q@researchgpu05.grid.gsb
# Strategist on pokemon_red: goal "obtain your first Pokemon".
# Runs the control (with notebook) then the ablation (without), so the pair is comparable.
#
# Submit:
#   qsub -l gpu=1,h_vmem=80G,h_rt=86400 -j y -cwd scripts/run_strategist_pokemon_red.sh

cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate
export HF_HOME="/scratch/af3698/huggingface_cache"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-required-using-vllm}"

MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
GAME="pokemon_red"
GOAL="Obtain your first Pokemon"
EXECUTOR="single_actions"
MAX_TASKS=8
MAX_STEPS=175

echo "=== GPU state at job start ==="
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader

# --- Start vLLM ---
VLLM_PID=""
VLLM_STARTED=0

# GPU_PIN lets concurrent jobs each claim a distinct card. Without it, jobs launched
# together all probe GPU 0 first, see it free (the previous job has not finished
# allocating yet), and collide. The grid leaves CUDA_VISIBLE_DEVICES unset on this
# cluster, so choosing the card is ours to do.
# GPU_PIN is a PREFERENCE, not a restriction: try that card first, then fall back to the
# rest. As a hard restriction it killed two jobs outright the moment another user occupied
# the pinned cards — pinning exists to stop concurrent jobs colliding on GPU 0, not to make
# a run fail when its one card is busy.
GPU_CANDIDATES="${GPU_PIN:+$GPU_PIN }0 1 2 3 4 5 6 7"
for GPU_IDX in $GPU_CANDIDATES; do
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

# --- MEASURED: strategist on benchmark tasks, graded by each task's tracker ---
#
# Equal-budget comparison against the bare executor arms in run_benchmark.py:
#   bare executor  : 1 attempt  x 175 steps
#   strategist     : 3 attempts x  60 steps = 180 steps, with memory between attempts
# so any difference is the planning layer, not a bigger step budget.
#
# TASKS is the scorable subset: the 34-of-49 tasks whose success-target .npy is missing
# from this machine can never fire their tracker, so including them would score every arm
# zero on them and dilute the comparison.
TASKS="${TASKS:-2}"
STRAT_TASKS=3
STRAT_STEPS=60

for T in $TASKS; do
  for NB in true false; do
    echo ""
    echo "=== benchmark task $T | use_notebook=$NB ==="
    python run_strategist.py \
      --game "$GAME" \
      --benchmark_task "$T" \
      --strategist tracker \
      --executor "$EXECUTOR" \
      --executor_vlm_model "$MODEL" \
      --executor_vlm_kind vllm \
      --max_tasks "$STRAT_TASKS" \
      --max_steps_per_task "$STRAT_STEPS" \
      --use_notebook "$NB" \
      --save_video false \
      --extra_name measured \
      --verbose
    echo "task $T notebook=$NB exit: $?"
  done
done

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM stopped."
echo ""
echo "=== DONE ==="
