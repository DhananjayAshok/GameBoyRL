#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args
declare -A ARGS
ARGS["executor_vlm_model"]="google/gemini-3.1-pro-preview"
ARGS["executor_vlm_kind"]="openrouter"   # use "none" for absent optionals, never ""
ARGS["max_steps"]="2"
ARGS["max_resets"]="3"
ARGS["random_sample"]="2"

REQUIRED_ARGS=("game")

# OPTIONAL: merge shared args from utils.sh (do this BEFORE ALLOWED_FLAGS)
populate_common_optional_training_args ARGS
populate_common_required_training_args REQUIRED_ARGS

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

# Put your script code below:

model_name="google/gemini-3.1-pro-preview"
model_kind="openrouter"

game="pokemon_red"
max_resets=3
max_steps=30


common=python run_benchmark_zeroshot.py --game ${ARGS["game"]} --save_video True --max_resets ${ARGS["max_resets"]} --max_steps ${ARGS["max_steps"]} --executor_vlm_model ${ARGS["executor_vlm_model"]} --executor_vlm_kind ${ARGS["executor_vlm_kind"]} 

eval $common --random_sample ${ARGS["random_sample"]} --verbose

eval $common