#!/usr/bin/env bash

source scripts/utils.sh

# Define Defaults for default_rl.sh
declare -A ARGS
ARGS["algorithm"]="dqn"
ARGS["gamma"]="0.99"
ARGS["similarity_metric"]="cosine"
ARGS["observation_embedder"]="random_patch"
ARGS["embedder_load_path"]="none"
ARGS["curiosity_module"]="embedbuffer"
ARGS["max_steps"]=50
ARGS["timesteps"]=1000000
ARGS["controller"]="low_level"
# Script specific defaults
ARGS["n_agents"]=10

# Temporarily hardcode game for testing
ARGS["game"]="pokemon_red"
ARGS["init_state"]="default"

# Define Required Keys
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
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

replay_buffer_save_folder=${ARGS["init_state"]}

## Set up functions for iterative training


function get_local_exp_name() {
    local buffer_load_path="$1"
    local exp_name=$(get_exp_name_full --algorithm ${ARGS["algorithm"]} --timesteps ${ARGS["timesteps"]} --gamma ${ARGS["gamma"]} --observation_embedder ${ARGS["observation_embedder"]} --embedder_load_path ${ARGS["observation_embedder"]} --curiosity_module ${ARGS["curiosity_module"]} --similarity_metric ${ARGS["similarity_metric"]}  --game ${ARGS["game"]} --env default --init_state ${ARGS["init_state"]} --controller ${ARGS["controller"]} --max_steps ${ARGS["max_steps"]} --buffer_load_path $buffer_load_path )
    if [-z "$exp_name" ]; then
        return 1
    fi
    echo "$exp_name"
}

prev_buffer_load_path="none"

function get_true_replay_buffer_save_folder() {
    local exp_name=$(get_local_exp_name $prev_buffer_load_path)
    if [-z "$exp_name" ]; then
        echo "Error: Could not determine experiment name for buffer load path '$prev_buffer_load_path'."
        return 1
    fi
    echo $replay_buffer_save_folder/$exp_name
}

log_folder="../iterative_agents/${ARGS["game"]}/${ARGS["init_state"]}/"
all_common_args="--gamma ${ARGS["gamma"]} --similarity_metric ${ARGS["similarity_metric"]} --observation_embedder ${ARGS["observation_embedder"]} --embedder_load_path ${ARGS["observation_embedder"]} --curiosity_module ${ARGS["curiosity_module"]} --max_steps ${ARGS["max_steps"]} --timesteps ${ARGS["timesteps"]} --controller ${ARGS["controller"]} --replay_buffer_save_folder $replay_buffer_save_folder --game ${ARGS["game"]} --env default --init_state ${ARGS["init_state"]} --log_folder $log_folder"

function call_agent(){
    local buffer_load_path="$1"
    local buffer_save_path="$2"
    bash scripts/default_rl.sh --algorithm ${ARGS["algorithm"]} --buffer_save_path $buffer_save_path --buffer_load_path $buffer_load_path $all_common_args
}

function train_world_model(){
    local buffer_load_path="$1"
    local buffer_save_path="$2"
    local latest_replay_buffer_folder=$(get_true_replay_buffer_save_folder)
    if [-z "$latest_replay_buffer_folder" ]; then
        echo "Error: Could not determine latest replay buffer folder for buffer load path '$buffer_load_path'."
        return 1
    fi
    bash scripts/train_world_model.sh --latest_replay_buffer_folder $latest_replay_buffer_folder --buffer_save_path $buffer_save_path --buffer_load_path $buffer_load_path --controller ${ARGS["controller"]} --game ${ARGS["game"]} --observation_embedder ${ARGS["observation_embedder"]} --embedder_load_path ${ARGS["embedder_load_path"]} 
}

## Execution starts here


buffer_save_path=${ARGS["init_state"]}
prev_buffer_save_path=$buffer_save_path

# First, run a random agent to populate the curiosity module buffer

bash scripts/default_rl.sh --algorithm random --buffer_save_path $buffer_save_path $all_common_args

if [ "${ARGS["curiosity_module"]}" == "world_model" ]; then
    train_world_model $prev_buffer_load_path $buffer_save_path    
fi

prev_buffer_load_path=$prev_buffer_save_path
buffer_save_path="${ARGS["init_state"]}_${ARGS["algorithm"]}_agent_1"

# Then, run the actual agent iteratively, updating the world model each time if needed
for ((i=0; i<${ARGS["n_agents"]}; i++)); do
    echo "Starting iteration $i with buffer load path '$prev_buffer_load_path' and buffer save path '$buffer_save_path'"

    call_agent $prev_buffer_load_path $buffer_save_path

    # Don't train world model on the last iteration since we won't be using the buffer again
    if [ $i -lt $((${ARGS["n_agents"]}-1)) ]; then
        if [ "${ARGS["curiosity_module"]}" == "world_model" ]; then
            train_world_model $prev_buffer_load_path $buffer_save_path    
        fi
    fi

    prev_buffer_load_path=$buffer_save_path
    buffer_save_path="${ARGS["init_state"]}_${ARGS["algorithm"]}_agent_$((i+2))"
done
