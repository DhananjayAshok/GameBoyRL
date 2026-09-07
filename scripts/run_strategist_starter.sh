#!/bin/bash
#$ -q gpu.q@researchgpu05.grid.gsb
# The first run whose START STATE MATCHES THE GOAL. Every earlier run began from
# pokemon_red's `default` state — Viridian City, no starter, Oak's lab several screens
# south — so "obtain your first Pokemon" was unreachable inside a task budget, and the
# failures said nothing about the strategist.
#
# `starter` puts the player inside Oak's Lab beside the three Poke Balls. The goal is a
# few presses away, so a failure here is genuinely attributable to the planner or the
# executor rather than to map distance.
#
# Submit:
#   qsub -l gpu=1,h_vmem=80G,h_rt=86400 -j y -cwd scripts/run_strategist_starter.sh

cd /user/af3698/GameBoyRL
source setup/.venv/bin/activate
export HF_HOME="/scratch/af3698/huggingface_cache"
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-required-using-vllm}"

MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
GAME="pokemon_red"
GOAL="Obtain your first Pokemon"
EXECUTOR="single_actions"
MAX_TASKS=6       # the goal is a few presses away from this state
MAX_STEPS=60      # enough to walk to a ball, press A, and confirm

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
echo "=== STRATEGIST from Oak's Lab (init_state=starter) ==="
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
  --extra_name starterstate \
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
