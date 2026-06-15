#!/bin/bash
#
# Usage: 
#   bash scripts/serve_vllm.sh [VLLM_ARGS...]
#
# Description:
#   A blocking wrapper around 'vllm serve'. It launches the server in the 
#   background, transparently forwards all command-line arguments to vLLM, 
#   and holds the terminal execution until the server health check passes 
#   or the process crashes.
#
# Examples:
#   bash scripts/serve_vllm.sh --model facebook/opt-125m
#   bash scripts/serve_vllm.sh --model meta-llama/Meta-Llama-3-8B-Instruct --port 8085 --tensor-parallel-size 2
#
# Outputs:
#   vllm.pid         - Stores the active background process ID (used by stop_vllm.sh)
#   vllm_server.log  - Standard output and error logs from the server
#
source setup/.venv/bin/activate
module load gcc/13.3.0 cuda/12.6.3
export TRITON_CACHE_DIR=/dev/shm/${USER}_triton_cache/
PORT=8000
LOG_FILE="vllm_server.log"
TIMEOUT=900 # 15 minutes timeout for server startup  

# Extract the port if provided in the args, otherwise fallback to 8000
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--port" ]]; then
        PORT="${ARGS[i+1]}"
    elif [[ "${ARGS[i]}" =~ --port=(.*) ]]; then
        PORT="${BASH_REMATCH[1]}"
    fi
done

echo "Launching vLLM on port $PORT..."

# Start vLLM in the background and pass ALL arguments through
nohup vllm serve "$@" > "$LOG_FILE" 2>&1 &
VLLM_PID=$!

# Block until the health endpoint responds or the process crashes
END_TIME=$((SECONDS + TIMEOUT))
while [ $SECONDS -lt $END_TIME ]; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "❌ ERROR: vLLM process crashed. Check $LOG_FILE"
        exit 1
    fi

    if curl -s -f http://localhost:${PORT}/health > /dev/null; then
        echo "✅ SUCCESS: vLLM is healthy!"
        echo $VLLM_PID > vllm.pid
        exit 0
    fi

    sleep 2
done

echo "❌ ERROR: Timed out waiting for vLLM to start."
kill -9 $VLLM_PID
exit 1