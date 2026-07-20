#!/usr/bin/env bash
# Shows what the VLM was actually fine-tuned on: dataset size and per-task imbalance, an
# audit of the training targets (unparseable and non-executable Action lines), which prompt
# blocks appear in training inputs, leakage checks across the train/validation split, and
# --n_samples random rows verbatim with their images. Writes
# <results_dir>/debug/<game>/dataset/report.md.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array DEBUG_MODEL_ESSENTIALS REQUIRED_ARGS
populate_dict DEBUG_MODEL_DEFAULTS ARGS

ARGS["extra"]="zeroshot_with_curiosity"
ARGS["n_samples"]=50
ARGS["seed"]=0

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

group_flags=$(debug_group_flags ARGS)

python debug.py $group_flags dataset \
    --model_name "${ARGS["model_name"]}" \
    --extra "${ARGS["extra"]}" \
    --n_samples "${ARGS["n_samples"]}" \
    --seed "${ARGS["seed"]}" || exit 1
