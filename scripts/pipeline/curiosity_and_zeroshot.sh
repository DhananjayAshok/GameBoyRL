#!/usr/bin/env bash
# Single init_state version of curiosity_and_zeroshot_all.sh: proposes and attempts.
# The curiosity annotation is checked for but is not used as a proposal prior — see
# curiosity_and_zeroshot_all.sh for why that arm was dropped.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array PROPOSE_AND_ATTEMPT_ESSENTIALS REQUIRED_ARGS
populate_dict PROPOSE_AND_ATTEMPT_DEFAULTS ARGS

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

model_save_name="${ARGS["model_name"]##*/}"
curiosity_dir="$storage_dir/proposed_tasks/${ARGS["game"]}/${model_save_name}/curiosity/${ARGS["run_name"]}"

if [[ ! -f "$curiosity_dir/trajectory_annotation.json" ]]; then
    echo "Error: trajectory_annotation.json not found at $curiosity_dir. Run infer_tasks first."
    exit 1
fi
if [[ ! -f "$curiosity_dir/trajectory_annotation.pkl" ]]; then
    echo "Error: trajectory_annotation.pkl not found at $curiosity_dir. Run infer_tasks first."
    exit 1
fi

# Zeroshot leg for this single init_state: propose, then attempt.
# The curiosity annotation checked for above is this game's curiosity vertical output; it
# is not consumed as a proposal prior (see curiosity_and_zeroshot_all.sh for why), so
# there is only one leg here.
ARGS["propose_only"]="false"
flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
bash scripts/pipeline/propose_and_attempt.sh $flags || exit 1
