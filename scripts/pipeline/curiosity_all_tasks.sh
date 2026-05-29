#!/usr/bin/env bash
# Batch version of curiosity_tasks: loops over all init_states for a game,
# running create_traj then infer_tasks for each. Regenerates TRAIN_STATES
# before iterating.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
python scripts/python/create_task_dictionary.py
source scripts/core/all_train_states.sh

declare -A ARGS
REQUIRED_ARGS=()

populate_array VLM_ESSENTIALS REQUIRED_ARGS
REQUIRED_ARGS+=("run_name")
populate_dict CURIOSITY_TASKS_DEFAULTS ARGS

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

if [[ "${ARGS["run_name"]}" == "none" ]]; then
    echo "Error: --run_name cannot be 'none'."; usage
fi

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

_do_guidance=false
[[ "${ARGS["do_guidance_and_practice"]}" == "true" || "${ARGS["do_guidance_and_practice"]}" == "yes" || "${ARGS["do_guidance_and_practice"]}" == "y" || "${ARGS["do_guidance_and_practice"]}" == "t" ]] && _do_guidance=true

game="${ARGS["game"]}"
IFS=',' read -ra init_states_arr <<< "${TRAIN_STATES[$game]}"
for init_state in "${init_states_arr[@]}"; do
    ARGS["init_state"]=$init_state

    create_flags=$(args_to_flags_subset ARGS CREATE_TRAJ_ARG_KEYS)
    bash scripts/rl/create_traj.sh $create_flags || exit 1

    init_state_group="${ARGS["init_state_group"]}"
    if [[ "$init_state_group" == "none" ]]; then
        init_state_group="$init_state"
    fi
    ARGS["trajectory_path"]="$storage_dir/grouped_trajectories/${game}/${ARGS["run_name"]}/${init_state_group}/grouped_global_high_reward_trajectories.pkl"

    infer_flags=$(args_to_flags_subset ARGS INFER_TASKS_ARG_KEYS)
    bash scripts/vlm/infer_tasks.sh $infer_flags || exit 1
done

model_save_name="${ARGS["model_name"]##*/}"
gp_trajectory_path="$storage_dir/proposed_tasks/${game}/${model_save_name}/curiosity/${ARGS["run_name"]}/trajectory_annotation"
if [[ "$_do_guidance" == "true" ]]; then
    ARGS["trajectory_path"]="$gp_trajectory_path"
    guidance_flags=$(args_to_flags_subset ARGS GUIDANCE_AND_PRACTICE_ARG_KEYS)
    bash scripts/pipeline/guidance_and_practice.sh $guidance_flags || exit 1
else
    echo "trajectory_path for guidance_and_practice: $gp_trajectory_path"
fi
