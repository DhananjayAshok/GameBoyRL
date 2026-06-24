#!/usr/bin/env bash
# Batch version of propose_and_attempt: proposes zero-shot tasks for every
# init_state via propose_zeroshot, then runs attempt_tasks once on the full
# combined tasks file. tasks_path is derived from game, model_name, and extra.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
python scripts/python/create_task_dictionary.py || { echo "Could not regenerate train states"; exit 1; }
source scripts/core/all_train_states.sh

declare -A ARGS
REQUIRED_ARGS=()

populate_array VLM_ESSENTIALS REQUIRED_ARGS
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

_propose_only=false
_attempt_only=false
[[ "${ARGS["propose_only"]}" == "true" || "${ARGS["propose_only"]}" == "yes" || "${ARGS["propose_only"]}" == "y" || "${ARGS["propose_only"]}" == "t" ]] && _propose_only=true
[[ "${ARGS["attempt_only"]}" == "true" || "${ARGS["attempt_only"]}" == "yes" || "${ARGS["attempt_only"]}" == "y" || "${ARGS["attempt_only"]}" == "t" ]] && _attempt_only=true
_do_guidance=false
[[ "${ARGS["do_guidance_and_practice"]}" == "true" || "${ARGS["do_guidance_and_practice"]}" == "yes" || "${ARGS["do_guidance_and_practice"]}" == "y" || "${ARGS["do_guidance_and_practice"]}" == "t" ]] && _do_guidance=true

model_save_name="${ARGS["model_name"]##*/}"
game="${ARGS["game"]}"
base="$storage_dir/proposed_tasks/${game}/${model_save_name}/zeroshot"
case "${ARGS["extra"]}" in
    none)                    tasks_file="$base/zeroshot_tasks.jsonl" ;;
    zeroshot)                tasks_file="$base/zeroshot_tasks_prior_zeroshot.jsonl" ;;
    curiosity)               tasks_file="$base/zeroshot_tasks_prior_curiosity.jsonl" ;;
    zeroshot_with_curiosity) tasks_file="$base/zeroshot_tasks_prior_zeroshot_with_curiosity.jsonl" ;;
    *) echo "Error: unknown extra value '${ARGS["extra"]}'"; exit 1 ;;
esac

if [[ "$_attempt_only" == "false" ]]; then
    ARGS["init_states"]="${TRAIN_STATES[$game]}"
    propose_flags=$(args_to_flags_subset ARGS PROPOSE_ZEROSHOT_ARG_KEYS)
    bash scripts/vlm/propose_zeroshot.sh $propose_flags || exit 1
fi

if [[ "$_propose_only" == "false" ]]; then
    if [[ ! -f "$tasks_file" ]]; then
        echo "Error: tasks file not found at $tasks_file. Run proposal phase first."
        exit 1
    fi
    ARGS["tasks_path"]="$tasks_file"
    attempt_flags=$(args_to_flags_subset ARGS ATTEMPT_TASKS_ARG_KEYS)
    bash scripts/vlm/attempt_tasks.sh $attempt_flags || exit 1

    if [[ "$_do_guidance" == "true" ]]; then
        ARGS["trajectory_path"]="${tasks_file%.jsonl}_${ARGS["executor"]}_attempts/success_trajectories"
        guidance_flags=$(args_to_flags_subset ARGS GUIDANCE_AND_PRACTICE_ARG_KEYS)
        bash scripts/pipeline/guidance_and_practice.sh $guidance_flags || exit 1
    else
        echo "trajectory_path for guidance_and_practice: ${tasks_file%.jsonl}_${ARGS["executor"]}_attempts/success_trajectories"
    fi
fi
