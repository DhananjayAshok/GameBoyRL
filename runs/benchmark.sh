#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args
declare -A ARGS
ARGS["executor_vlm_model"]="Qwen/Qwen3-VL-8B-Instruct"   # use "none" for absent optionals, never ""
ARGS["executor_vlm_kind"]="huggingface"   # use "none" for absent optionals, never ""
ARGS["max_steps"]="20"
ARGS["max_resets"]="3"
ARGS["random_sample"]="2"
ARGS["regenerate"]="false"

REQUIRED_ARGS=("game")


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

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

regenerate_flag=""
if [[ "${ARGS["regenerate"]}" == "true" ]]; then regenerate_flag="--regenerate"; fi

common="python run_benchmark_zeroshot.py --game ${ARGS["game"]} --save_video True --max_resets ${ARGS["max_resets"]} --max_steps ${ARGS["max_steps"]} --executor_vlm_model ${ARGS["executor_vlm_model"]} --executor_vlm_kind ${ARGS["executor_vlm_kind"]} $regenerate_flag"

$common --random_sample ${ARGS["random_sample"]} --verbose

$common