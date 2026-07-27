#!/bin/bash
#
# Usage:
#   bash scripts/core/stop_vllm.sh [PORT or --port PORT]
#
# Description:
#   Counterpart to serve_vllm.sh: stops the vLLM server on a given port.
#
#   1. If ~/vllm_scripts/stop_vllm.sh exists, pass through to it (the author's
#      cluster-specific teardown — see the note in serve_vllm.sh).
#
#   2. Otherwise, fall back to the generic path below: read the host+port-keyed
#      pid file that serve_vllm.sh's generic path wrote to $VLLM_STATE_DIR
#      (default ~/vllm_state), SIGTERM the process group, and SIGKILL it if the
#      grace period expires. Must run on the same node as the server, but works
#      from any cwd.
#

# --- Path 1: cluster-specific wrapper, if present ---
if [ -f "$HOME/vllm_scripts/stop_vllm.sh" ]; then
    echo "Found ~/vllm_scripts; routing through stop_vllm.sh"
    exec bash "$HOME/vllm_scripts/stop_vllm.sh" "$@"
fi

# --- Path 2: generic stop via the pid file serve_vllm.sh wrote ---
TIMEOUT=15
PORT=8000   # Default fallback port

# Parse port from arguments (accepts raw number or --port flag)
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[i]}" == "--port" ]]; then
        PORT="${ARGS[i+1]}"
    elif [[ "${ARGS[i]}" =~ --port=(.*) ]]; then
        PORT="${BASH_REMATCH[1]}"
    elif [[ "${ARGS[i]}" =~ ^[0-9]+$ ]]; then
        PORT="${ARGS[i]}"
    fi
done

# Resolve the same host+port key serve_vllm.sh wrote, so no state has to be
# passed between the two scripts and cwd doesn't matter.
STATE_DIR="${VLLM_STATE_DIR:-$HOME/vllm_state}"
HOST=$(hostname -s)
PID_FILE="$STATE_DIR/vllm_${HOST}_${PORT}.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️ No tracking file found at '$PID_FILE'. Is a server running on port $PORT on host $HOST?"
    exit 1
fi

VLLM_PID=$(cat "$PID_FILE")
echo "Using tracking file: $PID_FILE"

if kill -0 "$VLLM_PID" 2>/dev/null; then
    # Resolve the process group so we signal the whole vLLM tree (launcher +
    # EngineCore workers), not just the launcher. Otherwise workers orphan and
    # keep holding VRAM. Fall back to the bare PID if the group can't be read,
    # or if it matches our own group (guards against killing an interactive shell).
    PGID=$(ps -o pgid= -p "$VLLM_PID" | tr -d ' ')
    MYPGID=$(ps -o pgid= -p $$ | tr -d ' ')
    if [ -n "$PGID" ] && [ "$PGID" != "$MYPGID" ]; then
        TARGET="-$PGID"
    else
        TARGET="$VLLM_PID"
    fi

    echo "Stopping vLLM server on port $PORT gracefully (PID: $VLLM_PID, group: ${PGID:-n/a})..."
    kill -TERM "$TARGET" 2>/dev/null

    ELAPSED=0
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "✅ vLLM server on port $PORT stopped successfully."
            rm -f "$PID_FILE"
            exit 0
        fi
        sleep 1
        ((ELAPSED++))
    done

    echo "⚠️ Server on port $PORT didn't stop within $TIMEOUT seconds. Forcing shutdown..."
    kill -KILL "$TARGET" 2>/dev/null
    echo "☠️ vLLM server force-killed."
else
    echo "ℹ️ The vLLM process ($VLLM_PID) on port $PORT was already dead. Removing $PID_FILE"
fi

rm -f "$PID_FILE"
