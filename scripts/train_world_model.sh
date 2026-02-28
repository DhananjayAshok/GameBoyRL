#!/usr/bin/env bash

source scripts/utils.sh

# Define Defaults
declare -A ARGS
ARGS["observation_embedder"]="random_patch"
ARGS["embedder_load_path"]="none"
ARGS["latest_replay_buffer_folder"]="none"
ARGS["buffer_save_path"]="none"
ARGS["buffer_load_path"]="none"

# Temporarily hardcode game for testing
ARGS["controller"]="low_level"
ARGS["game"]="pokemon_red"

# Define Required Keys
#REQUIRED_ARGS=("game" "env")
REQUIRED_ARGS=() # temporarily make all optional for testing

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
echo "Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done


cd cleanrl

# Logic here:

train_env_id=$(get_env_id --game ${ARGS["game"]} --env default --init_state default --controller ${ARGS["controller"]} --max_steps 10)
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
    extra_arg_part+="--buffer_save_path $storage_dir/world_model/${ARGS["game"]}/${ARGS["buffer_save_path"]} "
fi

if [[ "${ARGS["buffer_load_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_load_path $storage_dir/world_model/${ARGS["game"]}/${ARGS["buffer_load_path"]} "
fi
if [[ "${ARGS["embedder_load_path"]}" != "none" ]]; then
    extra_arg_part+="--embedder_load_path $storage_dir/${ARGS["observation_embedder"]}/${ARGS["game"]}/${ARGS["embedder_load_path"]} "
fi


echo "Training World Model:"

echo python cleanrl_utils/train_world_model.py --seed 1 --env-id $env_id \
    --track --wandb-project-name $WANDB_PROJECT \
    --observation-embedder ${ARGS["observation_embedder"]} $extra_arg_part