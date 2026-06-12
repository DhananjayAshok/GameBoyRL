#!/bin/bash
#
# Usage:
#   bash scripts/stop_vllm.sh
#
# Description:
#   Gracefully shuts down the background vLLM server initiated by serve_vllm.sh.
#   It reads the process ID from 'vllm.pid', issues a SIGTERM to allow proper 
#   VRAM release, and monitors the exit. Falls back to a hard SIGKILL if the 
#   grace period expires.
#
# Inputs:
#   vllm.pid         - Expected file containing the target process ID.
#

# Configuration
TIMEOUT=15  # Max seconds to wait for a graceful shutdown
PID_FILE="vllm.pid"

# 1. Check if the PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "⚠️ No $PID_FILE found. Is the server even running?"
    exit 1
fi

VLLM_PID=$(cat "$PID_FILE")

# 2. Check if the process is actually alive
if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM server gracefully (PID: $VLLM_PID)..."
    kill "$VLLM_PID" # Sends SIGTERM
    
    # 3. Dynamic wait loop based on the TIMEOUT parameter
    ELAPSED=0
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "✅ vLLM server stopped successfully."
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
        ((ELAPSED++))
    done

    # 4. Force kill if timeout is reached
    echo "⚠️ Server didn't stop within $TIMEOUT seconds. Forcing shutdown..."
    kill -9 "$VLLM_PID"
    echo "☠️ vLLM server force-killed."
else
    echo "ℹ️ The vLLM process ($VLLM_PID) was already dead."
fi

# Clean up the artifact
rm -f "$PID_FILE"