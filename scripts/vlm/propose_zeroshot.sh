#!/usr/bin/env bash
# Calls the VLM to propose candidate tasks for a single game/init_state without
# any prior trajectory data (zero-shot). Results are saved to the project's task
# store and can be augmented with extra examples via --extra and --extra_k. Use
# propose_all_zeroshot.sh to run across all init_states for a game.

source scripts/core/utils.sh

declare -A ARGS
REQUIRED_ARGS=()

populate_dict PROPOSE_ZEROSHOT_DEFAULTS ARGS
populate_array PROPOSE_ZEROSHOT_ESSENTIALS REQUIRED_ARGS

ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")

USAGE_STR="Usage: $0"

for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done

for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done

function usage() {
    echo "$USAGE_STR"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then
                    VALID=true
                    break
                fi
            done
            if [ "$VALID" = false ]; then
                echo "Error: Unknown flag --$FLAG"
                usage
            fi
            ARGS["$FLAG"]="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Argument --$req is required."
        FAILED=true
    fi
done

if [ "$FAILED" = true ]; then usage; fi

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

game="${ARGS["game"]}"
model_name="${ARGS["model_name"]}"
vlm_kind="${ARGS["vlm_kind"]}"
init_state="${ARGS["init_state"]}"
max_new_tokens="${ARGS["max_new_tokens"]}"
extra="${ARGS["extra"]}"
extra_k="${ARGS["extra_k"]}"

if [[ "${ARGS["overwrite"]}" == "true" || "${ARGS["overwrite"]}" == "yes" || "${ARGS["overwrite"]}" == "y" ]]; then
    overwrite_flag="--overwrite"
else
    overwrite_flag=""
fi

if [[ "${ARGS["verbose"]}" == "true" || "${ARGS["verbose"]}" == "yes" || "${ARGS["verbose"]}" == "y" ]]; then
    verbose_flag="--verbose"
else
    verbose_flag=""
fi

if [[ "$extra" == "none" ]]; then
    extra_flag=""
else
    extra_flag="--extra $extra --extra_k $extra_k"
fi

python vlm.py --game $game --model_name $model_name --vlm_kind $vlm_kind \
    --max_new_tokens $max_new_tokens $overwrite_flag $verbose_flag \
    propose_tasks_zeroshot --init_state $init_state --run_name "${ARGS["run_name"]}" $extra_flag
