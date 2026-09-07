#!/bin/bash
#$ -q gpu.q@researchgpu05.grid.gsb
# LONG-HORIZON: the actual goal — play the game as far as possible from Oak's Lab.
#
# Scored on a PROGRESS CURVE (badges, unique locations, starter, walk steps), not on the
# binary "did you finish the game", which would read 0 for both arms and distinguish
# nothing. The 12-task benchmark sweep measured the strategist where it adds least value:
# single-screen tasks need no decomposition. This measures it where it was designed to work.
#
# One arm per job so the two run concurrently on separate GPUs; ARM and GPU_PIN select.
#
# Submit:
#   qsub -l gpu=1,h_vmem=80G,h_rt=86400 -j y -cwd scripts/run_strategist_pokemon_red.sh

cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate
export HF_HOME="/scratch/af3698/huggingface_cache"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-required-using-vllm}"

MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
GAME="${GAME:-pokemon_red}"   # pokemon_red | pokemon_prism | pokemon_brown
GOAL="${GOAL:-Play through the game as far as you can: obtain a Pokemon, explore new towns and routes, and earn gym badges}"
EXECUTOR="single_actions"
MAX_TASKS="${MAX_TASKS:-14}"   # >10 so compression fires; 14 x ~66min ~= 15h, inside the wall
MAX_STEPS="${MAX_STEPS:-175}"     # same per-attempt budget as the benchmark arms

echo "=== GPU state at job start ==="
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader

# --- Start vLLM ---
VLLM_PID=""
VLLM_STARTED=0

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

# --- one arm, selected by $ARM ---
ARM="${ARM:-true}"          # true = control (notebook), false = ablation
echo ""
echo "=== PLAYTHROUGH $GAME | strategist=${STRAT:-subgoal} | notebook=$ARM | $MAX_TASKS x $MAX_STEPS ==="
python run_strategist.py \
  --game "$GAME" \
  --goal "$GOAL" \
  --executor "$EXECUTOR" \
  --executor_vlm_model "$MODEL" \
  --executor_vlm_kind vllm \
  --max_tasks "$MAX_TASKS" \
  --max_steps_per_task "$MAX_STEPS" \
  --strategist "${STRAT:-subgoal}" \
  --use_notebook "$ARM" \
  --verify_goal false \
  --stop_on_goal false \
  --save_video true \
  --extra_name longhorizon \
  --verbose
RUN_EXIT=$?
echo "arm=$ARM exit: $RUN_EXIT"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM stopped."
echo ""
echo "=== DONE ==="
exit $RUN_EXIT
