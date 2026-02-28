#!/usr/bin/env bash

source scripts/utils.sh

# Define Defaults
declare -A ARGS
ARGS["algorithm"]="sac"
ARGS["gamma"]="0.99"
ARGS["similarity_metric"]="cosine"
ARGS["observation_embedder"]="random_patch"
ARGS["embedder_load_path"]="none"
ARGS["curiosity_module"]="embedbuffer"
ARGS["max_steps"]=200
ARGS["timesteps"]=500000
ARGS["test_env"]=""
ARGS["test_init_state"]=""
ARGS["replay_buffer_save_folder"]="none"
ARGS["buffer_save_path"]="none"
ARGS["buffer_load_path"]="none"
ARGS["train_only"]=""
ARGS["controller"]="low_level"
ARGS["log_folder"]=""
ARGS["env"]="default"

# Temporarily hardcode game for testing
ARGS["game"]="pokemon_red"
ARGS["init_state"]="default"

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

train_env_id=$(get_env_id --game ${ARGS["game"]} --env ${ARGS["env"]} --init_state ${ARGS["init_state"]} --controller ${ARGS["controller"]} --max_steps ${ARGS["max_steps"]})
if [[ -z "$train_env_id" ]]; then
    echo "Error: Failed to get train_env_id"
    exit 1
fi
train_env_id=$train_env_id-False

exp_name=$(get_exp_name_partial --algorithm ${ARGS["algorithm"]} --timesteps ${ARGS["timesteps"]} --gamma ${ARGS["gamma"]} --observation_embedder ${ARGS["observation_embedder"]} --embedder_load_path ${ARGS["observation_embedder"]} --curiosity_module ${ARGS["curiosity_module"]} --similarity_metric ${ARGS["similarity_metric"]} --buffer_load_path ${ARGS["buffer_load_path"]} )
if [[ -z "$exp_name" ]]; then
    echo "Error: Failed to generate exp_name"
    exit 1
fi
exp_name=$exp_name-$env_id

model_save_path="$storage_dir/models/$exp_name/"

extra_arg_part=""
if [[ "${ARGS["replay_buffer_save_folder"]}" != "none" ]]; then
    extra_arg_part+="--replay_buffer_save_folder $storage_dir/replay_buffers/${ARGS["game"]}/${ARGS["replay_buffer_save_folder"]} "
fi

if [[ "${ARGS["buffer_save_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_save_path $storage_dir/${ARGS["curiosity_module"]}/${ARGS["game"]}/${ARGS["buffer_save_path"]} "
fi

if [[ "${ARGS["buffer_load_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_load_path $storage_dir/${ARGS["curiosity_module"]}/${ARGS["game"]}/${ARGS["buffer_load_path"]} "
fi
if [[ "${ARGS["embedder_load_path"]}" != "none" ]]; then
    extra_arg_part+="--embedder_load_path $storage_dir/${ARGS["observation_embedder"]}/${ARGS["game"]}/${ARGS["embedder_load_path"]} "
fi

log_file="../$exp_name.out"
if [[ -n "${ARGS["log_folder"]}" ]]; then
    log_file="$storage_dir/logs/${ARGS["log_folder"]}/$exp_name.out"
    # make sure the folder exists
    mkdir -p "$(dirname "$log_file")"
fi

echo "Starting Experiment: $exp_name logging to $log_file"

python cleanrl/${ARGS["algorithm"]}_curiosity.py --exp_name $exp_name --seed 1 --gamma ${ARGS["gamma"]} --env-id $train_env_id --total-timesteps ${ARGS["timesteps"]} --track \
    --wandb-project-name $WANDB_PROJECT --model_save_path $model_save_path --capture_video --save_model \
    --observation-embedder ${ARGS["observation_embedder"]} --similarity_metric ${ARGS["similarity_metric"]} \
    --curiosity-module ${ARGS["curiosity_module"]} --reset-curiosity-module $extra_arg_part &> $log_file

# if test_env and test_init_state are empty, set them to train values:
if [[ -z "${ARGS["test_env"]}" ]]; then
    ARGS["test_env"]="${ARGS["env"]}"
fi
if [[ -z "${ARGS["test_init_state"]}" ]]; then
    ARGS["test_init_state"]="${ARGS["init_state"]}"
fi

cd ..

# if train_only is true, t, yes or y then exit here:
if [[ "${ARGS["train_only"]}" == "true" || "${ARGS["train_only"]}" == "yes" || "${ARGS["train_only"]}" == "y" ]]; then
    echo "Train only flag is set. Exiting after training."
    exit 0
fi


echo "Calling scripts/enjoy.sh --algorithm ${ARGS["algorithm"]} --exp_name $exp_name --env ${ARGS["test_env"]} --game ${ARGS["game"]} --init_state ${ARGS["test_init_state"]} --controller ${ARGS["controller"]} --max_steps ${ARGS["max_steps"]} --curiosity_module ${ARGS["curiosity_module"]} --observation_embedder ${ARGS["observation_embedder"]} --embedder_load_path ${ARGS["embedder_load_path"]} --similarity_metric ${ARGS["similarity_metric"]} --buffer_load_path ${ARGS["buffer_load_path"]} --buffer_save_path ${ARGS["buffer_save_path"]}"
bash scripts/enjoy.sh --algorithm ${ARGS["algorithm"]} --exp_name $exp_name --env ${ARGS["test_env"]} --game ${ARGS["game"]} --init_state ${ARGS["test_init_state"]} --controller ${ARGS["controller"]} --max_steps ${ARGS["max_steps"]} --curiosity_module ${ARGS["curiosity_module"]} --observation_embedder ${ARGS["observation_embedder"]} --embedder_load_path ${ARGS["embedder_load_path"]} --similarity_metric ${ARGS["similarity_metric"]} --buffer_load_path ${ARGS["buffer_load_path"]} --buffer_save_path ${ARGS["buffer_save_path"]}
