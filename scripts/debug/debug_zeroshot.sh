#!/usr/bin/env bash
# Reports zero-shot task proposal: for every init_state, the first frame the proposer saw,
# the tasks it proposed, and the benchmark tasks actually anchored to that state. Writes
# one <results_dir>/debug/<game>/zeroshot/report_<extra>.md per proposal variant.
#
# This is the only debug script that starts an emulator (the first frame is saved nowhere),
# so it needs the game's ROM. No GPU and no VLM.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array DEBUG_MODEL_ESSENTIALS REQUIRED_ARGS
populate_dict DEBUG_MODEL_DEFAULTS ARGS

ARGS["extra"]="all"
ARGS["controller_variant"]="low_level"

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

python debug.py $group_flags zeroshot \
    --model_name "${ARGS["model_name"]}" \
    --extra "${ARGS["extra"]}" \
    --controller_variant "${ARGS["controller_variant"]}" || exit 1
