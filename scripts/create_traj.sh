#init_states="default train_0 train_1 train_2 train_3 train_4"
#init_state_group="default" # args


#!/usr/bin/env bash

source scripts/utils.sh

declare -A ARGS
REQUIRED_ARGS=()

populate_dict CREATE_TRAJ_DEFAULTS ARGS
populate_array  CREATE_TRAJ_ESSENTIALS REQUIRED_ARGS


ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")

USAGE_STR="Usage: $0"

# Add Required to string
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done

# Add Optionals to string
for opt in "${!ARGS[@]}"; do
    # Only list if NOT in required (to avoid double listing)
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done

function usage() {
    echo "$USAGE_STR"
    exit 1
}

# 3. Parser
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            # Extract the name (remove the leading --)
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

# 4. Strict Validation
for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then
        echo "Error: Argument --$req is required."
        FAILED=true
    fi
done

if [ "$FAILED" = true ]; then usage; fi

# error out if run_name is none or unset
if [[ -z "${ARGS["run_name"]}" || "${ARGS["run_name"]}" == "none" ]]; then
    echo "Error: --run_name is required and cannot be 'none'."
    usage
fi

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

sweeping=false
if [ "${ARGS["sweep"]}" == "true" ]; then
    sweep_run_name="iterative_sweep"
    sweeping=true
else
    sweep_run_name="iterative_agent"
fi
init_state_group=${ARGS["init_state_group"]}
replay_buffer_save_folder=${ARGS["run_name"]}/${init_state_group}/$sweep_run_name/
echo "Setting replay_buffer_save_folder to $replay_buffer_save_folder for iterative training"



# if skip_training, skip the bottom
if [ "${ARGS["skip_training"]}" == "true" ]; then
    echo "Skipping training as per --skip_training flag."
else
    echo "Clearing replay buffer for iterative training..."
    rm -rf $storage_dir/replay_buffers/${ARGS["game"]}/$replay_buffer_save_folder/* # clear replay buffer to ensure we don't have old trajectories lying around.
    run_name=${ARGS["run_name"]}
    echo "Run name: $run_name | Setting model_dir to run_name for iterative training"
    ARGS["model_dir"]="${run_name}"

    init_states=${ARGS["init_states"]}
    init_state_group=${ARGS["init_state_group"]}
    IFS=',' read -ra init_states_arr <<< "$init_states"
    for init_state in "${init_states_arr[@]}"; do
        ARGS["init_state"]=$init_state
        argstring=$(args_to_flags_subset ARGS ITERATIVE_TRAINING_ARG_KEYS)    
        bash scripts/iterative_training.sh $argstring
    done
fi

#bash scripts/group_trajectories.sh --game ${ARGS["game"]} --replay_buffer_folder $replay_buffer_save_folder --save_path $storage_dir/grouped_trajectories/${ARGS["game"]}/${ARGS["run_name"]}/${init_state_group}/ --z_min ${ARGS["z_min"]}