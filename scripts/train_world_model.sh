#!/usr/bin/env bash

source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }

# Define Defaults
declare -A ARGS
ARGS["algorithm"]="sac"
ARGS["gamma"]="0.99"
ARGS["similarity_metric"]="cosine"
ARGS["observation_embedder"]="random_patch"
ARGS["curiosity_module"]="embedbuffer"
ARGS["max_steps"]=200
ARGS["timesteps"]=500000
ARGS["test_env"]=""
ARGS["test_init_state"]=""
ARGS["replay_buffer_save_folder"]=""
ARGS["buffer_save_path"]=""
ARGS["buffer_load_path"]=""
ARGS["train_only"]=""
ARGS["controller"]="low_level"
ARGS["log_folder"]=""
ARGS["env"]="default"

# Temporarily hardcode game for testing
ARGS["game"]="pokemon_red"
ARGS["init_state"]="none"

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
train_env_id="poke_worlds-${ARGS["game"]}-${ARGS["env"]}-${ARGS["init_state"]}-${ARGS["controller"]}-${ARGS["max_steps"]}-False"
exp_name="${ARGS["algorithm"]}-${ARGS["timesteps"]}-${ARGS["gamma"]}-${ARGS["observation_embedder"]}-${ARGS["curiosity_module"]}-${ARGS["similarity_metric"]}-${train_env_id}"
model_save_path="$storage_dir/models/$exp_name/"

buffer_arg_part=""
if [[ -n "${ARGS["replay_buffer_save_folder"]}" ]]; then
    buffer_arg_part+="--replay_buffer_save_folder $storage_dir/replay_buffers/$ARGS["game"]/${ARGS["replay_buffer_save_folder"]} "
fi

if [[ -n "${ARGS["buffer_save_path"]}" ]]; then
    buffer_arg_part+="--buffer_save_path $storage_dir/${ARGS["curiosity_module"]}/$ARGS["game"]/${ARGS["buffer_save_path"]} "
fi

if [[ -n "${ARGS["buffer_load_path"]}" ]]; then
    buffer_arg_part+="--buffer_load_path $storage_dir/${ARGS["curiosity_module"]}/$ARGS["game"]/${ARGS["buffer_load_path"]} "
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
    --curiosity-module ${ARGS["curiosity_module"]} --reset-curiosity-module $buffer_arg_part &> $log_file

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

buffer_arg_part=""
if [[ -n "${ARGS["buffer_load_path"]}" ]]; then
    buffer_arg_part+="--buffer_load_path ${ARGS["buffer_load_path"]} "
fi

if [[ -n "${ARGS["buffer_save_path"]}" ]]; then
    buffer_arg_part+="--buffer_save_path ${ARGS["buffer_save_path"]} "
fi


echo "Calling scripts/enjoy.sh --algorithm ${ARGS["algorithm"]} --exp_name $exp_name --env ${ARGS["test_env"]} --game ${ARGS["game"]} --init_state ${ARGS["test_init_state"]} --controller ${ARGS["controller"]} --max_steps ${ARGS["max_steps"]} --curiosity_module ${ARGS["curiosity_module"]} --observation_embedder ${ARGS["observation_embedder"]} --similarity_metric ${ARGS["similarity_metric"]} $buffer_arg_part"
bash scripts/enjoy.sh --algorithm ${ARGS["algorithm"]} --exp_name $exp_name --env ${ARGS["test_env"]} --game ${ARGS["game"]} --init_state ${ARGS["test_init_state"]} --controller ${ARGS["controller"]} --max_steps ${ARGS["max_steps"]} --curiosity_module ${ARGS["curiosity_module"]} --observation_embedder ${ARGS["observation_embedder"]} --similarity_metric ${ARGS["similarity_metric"]} $buffer_arg_part
