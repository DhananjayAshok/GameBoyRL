#!/usr/bin/env bash
# Reports the benchmark stage. Writes two markdowns under
# <results_dir>/debug/<game>/benchmark/:
#   episodes_<model>.md  every episode step by step, read from the archived supervisor
#                        report beside that episode's video (per model, unlike the per-call
#                        PNGs which are keyed on the executor class and overwritten by
#                        whichever run finished last)
#   comparison.md        paired base vs fine-tuned: success rates, subgoal fractions, and
#                        every task solved by one model but not the other

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array DEBUG_MODEL_ESSENTIALS REQUIRED_ARGS
populate_dict DEBUG_MODEL_DEFAULTS ARGS

ARGS["compare_model"]="none"
ARGS["bench_game"]="none"
ARGS["max_episodes"]=0
ARGS["supervisor"]="dummy"
ARGS["n_tasks"]="none"

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

if [[ "${ARGS["n_tasks"]}" != "none" ]]; then
    n_tasks_arg="--n_tasks ${ARGS["n_tasks"]}"
else
    n_tasks_arg=""
fi

python debug.py $group_flags benchmark \
    --model_name "${ARGS["model_name"]}" \
    --compare_model "${ARGS["compare_model"]}" \
    --bench_game "${ARGS["bench_game"]}" \
    --supervisor "${ARGS["supervisor"]}" \
    --max_episodes "${ARGS["max_episodes"]}" $n_tasks_arg || exit 1
