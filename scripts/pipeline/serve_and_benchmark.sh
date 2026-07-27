#!/usr/bin/env bash
# Serves a fine-tuned VLM checkpoint with vLLM, then benchmarks it on every game in
# its series (the train game + the shifted eval game(s)) for a single executor.
#
# The checkpoint path is derived the same way train_vlm.sh writes it:
#   $storage_dir/models/${game}-${run_name}/${model_name#*/}/final_checkpoint
# The server is started via scripts/core/serve_vllm.sh on the largest tensor-parallel
# size that fits the available GPUs (max of {8,4,2}), and always stopped on exit.
#
# The benchmark set is data-driven: the series is the GameBoyWorlds/benchmark/tests/
# <series>.csv whose `game` column contains --game; every unique game in that file is
# benchmarked (mirrors runs/benchmark_vllm.sh, but scoped to one series + one executor).

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

# Define defaults and required args.
declare -A ARGS
ARGS["max_steps"]=75
ARGS["regenerate"]=false
ARGS["port"]=8000
ARGS["tensor_parallel"]=none   # none => auto-pick from available GPUs
ARGS["mode"]="both"            # must match the --mode full.sh trained under
ARGS["checkpoint"]=none        # none => derive from game/run_name/mode; set to benchmark any checkpoint

REQUIRED_ARGS=("game" "model_name" "run_name" "executor")

# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

game="${ARGS["game"]}"
model_name="${ARGS["model_name"]}"
run_name="${ARGS["run_name"]}"
executor="${ARGS["executor"]}"
model_save_name="${model_name#*/}"   # strip any org/ prefix, matching train_vlm.sh

# Single source of truth for the name vLLM registers the model under AND the name the
# benchmark asks the vLLM API for. These MUST be identical or the benchmark 404s on the
# model. run_benchmark.py keys result files off this name
# (results/benchmark/<game>/<executor>_<name>.csv), so it must encode the *checkpoint's*
# identity — not just the base model — or a fine-tuned run would clobber the base model's
# results (both would be "gemma-4-31b-it"). Mirror the checkpoint dir (${game}-${run_name})
# so it is unique per checkpoint and self-describing across cross-game evals.
# The mode is part of the identity: full.sh trains under "${game}-${run_name}-${mode}", so two
# modes sharing a --run_name would otherwise resolve to the same checkpoint, the same served
# name and the same benchmark CSV, silently overwriting each other.
mode="${ARGS["mode"]}"
served_model_name="${model_save_name}-${game}-${run_name}-${mode}"

# --- Derive and validate the checkpoint path (mirrors train_vlm.sh output_dir) ---
# --checkpoint overrides the derivation so an arbitrary checkpoint can be benchmarked without
# reverse-engineering the naming; the served name still describes what is being served.
if [[ "${ARGS["checkpoint"]}" != "none" ]]; then
    checkpoint="${ARGS["checkpoint"]}"
    echo "Using explicit --checkpoint (skipping the game/run_name/mode derivation)"
else
    checkpoint="$storage_dir/models/${game}-${run_name}-${mode}/${model_save_name}/final_checkpoint"
fi
if [[ ! -d "$checkpoint" ]]; then
    echo "Error: checkpoint not found at $checkpoint"; exit 1
fi
echo "Checkpoint: $checkpoint"
echo "Served model name: $served_model_name"

# --- Resolve the game series -> the set of games to benchmark ---
# Find the series CSV whose `game` column (col 1) contains --game, then collect every
# unique game in that file (train game + shifted eval game(s)).
series_csv=""
for f in "$PROJECT_ROOT"/GameBoyWorlds/benchmark/tests/*.csv; do
    if awk -F',' -v g="$game" 'NR>1 && $1==g {found=1} END{exit !found}' "$f"; then
        series_csv="$f"; break
    fi
done
if [[ -z "$series_csv" ]]; then
    echo "Error: could not find a series CSV under GameBoyWorlds/benchmark/tests/ containing game '$game'"; exit 1
fi
mapfile -t series_games < <(awk -F',' 'NR>1 && $1!="" {print $1}' "$series_csv" | sort -u)
if [[ ${#series_games[@]} -eq 0 ]]; then
    echo "Error: no games found in series CSV $series_csv"; exit 1
fi
echo "Series: $(basename "$series_csv" .csv) -> benchmark games: ${series_games[*]}"

# --- Pick tensor-parallel size: largest of {8,4,2} that fits the available GPUs ---
if [[ "${ARGS["tensor_parallel"]}" != "none" ]]; then
    tp="${ARGS["tensor_parallel"]}"
else
    n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if   (( n_gpus >= 8 )); then tp=8
    elif (( n_gpus >= 4 )); then tp=4
    elif (( n_gpus >= 2 )); then tp=2
    elif (( n_gpus >= 1 )); then tp=1; echo "WARNING: only $n_gpus GPU available; using tensor-parallel-size 1"
    else echo "Error: no GPUs detected via nvidia-smi"; exit 1
    fi
fi
echo "GPUs available: ${n_gpus:-override} -> tensor-parallel-size: $tp"

# --- Serve the checkpoint. Guarantee the server is stopped on any exit. ---
# NOTE: serve_vllm.sh routes through ~/vllm_scripts if present (cluster-specific CUDA
# modules/venvs), otherwise launches `vllm serve` from the current environment directly.
# Either way it blocks until the server is healthy.
bash scripts/core/serve_vllm.sh "$checkpoint" \
    --served-model-name "$served_model_name" \
    --tensor-parallel-size "$tp" \
    --port "${ARGS["port"]}" || { echo "Error: vLLM failed to start"; exit 1; }

# NOTE: stop_vllm.sh routes through ~/vllm_scripts if present, else SIGTERMs the vllm
# process group its generic serve path recorded at launch. It must be passed the same
# port we served on, or it will look for the wrong tracking file.
trap 'echo "Stopping vLLM server..."; bash scripts/core/stop_vllm.sh "${ARGS["port"]}"' EXIT

# --- Benchmark every game in the series for the single requested executor ---
status=0
for bench_game in "${series_games[@]}"; do
    echo "Running benchmark for game: $bench_game (executor: $executor)"
    bash scripts/benchmark.sh \
        --game "$bench_game" \
        --executor "$executor" \
        --executor_vlm_model "$served_model_name" \
        --executor_vlm_kind vllm \
        --max_steps "${ARGS["max_steps"]}" \
        --regenerate "${ARGS["regenerate"]}" || { echo "Benchmark failed for $bench_game"; status=1; }
done

exit $status
