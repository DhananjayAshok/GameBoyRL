#!/usr/bin/env bash
# Merges the train/validation datasets of several practice dirs into one pair, tagging every
# row with the practice dir it came from (a `source` column). Used by
# curiosity_and_zeroshot_all.sh to combine the curiosity and zeroshot verticals into
# a single fine-tuning set.
#
# Images are referenced, not copied — the dataset CSVs hold absolute paths into each source's
# images/ dir, so the merged CSV points at the originals.
#
# NOTE: "none" is a legitimate value of --balance here (no downsampling), not the usual
# "argument absent" sentinel. There is no --overwrite: the merge always regenerates, since it
# costs seconds against upstream legs that take days, and a stale merge would be trained on.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array MERGE_PRACTICES_ESSENTIALS REQUIRED_ARGS
populate_dict MERGE_PRACTICES_DEFAULTS ARGS

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

python create_dataset.py merge_practices \
    --practice_paths "${ARGS["practice_paths"]}" \
    --save_path "${ARGS["save_path"]}" \
    --balance "${ARGS["balance"]}" \
    --seed "${ARGS["seed"]}" || exit 1
