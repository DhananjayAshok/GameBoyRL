#!/usr/bin/env bash
# Trains a world model on a game's replay buffer, producing a predictive curiosity
# module for use in subsequent RL training (curiosity_module=world_model). Called
# between agent iterations in iterative_training.sh to keep the world model fresh
# as the replay buffer grows.

source scripts/core/utils.sh

# Define Defaults
declare -A ARGS
REQUIRED_ARGS=()

populate_dict WORLD_MODEL_DEFAULTS ARGS
populate_dict WORLD_MODEL_ESSENTIALS REQUIRED_ARGS


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

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done



# Logic here:

env_arg_str="--game ${ARGS["game"]} --env default --init_state default --controller ${ARGS["controller"]} --max_steps 10"
train_env_id=$(python "$PROJECT_ROOT/python_funcs.py" strings env_id $env_arg_str)
if [[ -z "$train_env_id" ]]; then
    echo "Error: Failed to get train_env_id"
    exit 1
fi
env_id=$train_env_id-False

extra_arg_part=""
if [[ "${ARGS["latest_replay_buffer_folder"]}" != "none" ]]; then
    extra_arg_part+="--latest_replay_buffer_folder $storage_dir/replay_buffers/${ARGS["game"]}/${ARGS["latest_replay_buffer_folder"]} "
fi

if [[ "${ARGS["buffer_save_path"]}" != "none" ]]; then
    buffer_save_dir=$(path_of world_model_dir --game "${ARGS["game"]}" --run_name "${ARGS["buffer_save_path"]}")
    extra_arg_part+="--buffer_save_path $buffer_save_dir "
fi

if [[ "${ARGS["buffer_load_path"]}" != "none" ]]; then
    buffer_load_dir=$(path_of world_model_dir --game "${ARGS["game"]}" --run_name "${ARGS["buffer_load_path"]}")
    extra_arg_part+="--buffer_load_path $buffer_load_dir "
fi
if [[ "${ARGS["embedder_load_path"]}" != "none" ]]; then
    embedder_load_dir=$(path_of observation_embedder_dir --game "${ARGS["game"]}" --run_name "${ARGS["embedder_load_path"]}")
    extra_arg_part+="--embedder_load_path $embedder_load_dir "
fi


echo "Training World Model:"
cd cleanrl
python cleanrl_utils/train_world_model.py --seed 1 --env-id $env_id \
    --track --wandb-project-name $WANDB_PROJECT \
    --observation_embedder ${ARGS["observation_embedder"]} $extra_arg_part

cd ..