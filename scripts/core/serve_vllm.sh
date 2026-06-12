#!/bin/bash

PORT=8000
LOG_FILE="vllm_server.log"
TIMEOUT=300

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