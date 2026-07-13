#!/bin/bash
#$ -j y
nvidia-smi
echo "---"
nvidia-smi pmon -c 1 2>/dev/null | head -30
echo "---"
# Kill any of my own stale processes holding GPU memory
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  owner=$(ps -o user= -p $pid 2>/dev/null)
  if [ "$owner" = "af3698" ]; then
    echo "Found my stale GPU process: $pid ($(ps -o comm= -p $pid 2>/dev/null)), killing"
    kill -9 $pid 2>/dev/null && echo "  killed $pid" || echo "  could not kill $pid"
  else
    echo "GPU process $pid owned by: $owner (not killing)"
  fi
done
echo "---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null
