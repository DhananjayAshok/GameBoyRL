#!/usr/bin/env bash

source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }

# Define Defaults for default_rl.sh
declare -A ARGS
ARGS["algorithm"]="sac"
ARGS["gamma"]="0.99"
ARGS["similarity_metric"]="cosine"
ARGS["observation_embedder"]="random_patch"
ARGS["curiosity_module"]="embedbuffer"
ARGS["max_steps"]=200
ARGS["timesteps"]=500000
ARGS["controller"]="low_level"
# Script specific defaults
ARGS["n_agents"]=5

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


common_args="--gamma ${ARGS["gamma"]} --similarity_metric ${ARGS["similarity_metric"]} --observation-embedder ${ARGS["observation_embedder"]} --curiosity-module ${ARGS["curiosity_module"]} --max_steps ${ARGS["max_steps"]} --timesteps ${ARGS["timesteps"]} --controller ${ARGS["controller"]}"
replay_buffer_save_folder=${ARGS["init_state"]}
buffer_save_path=${ARGS["init_state"]}
bash scripts/default_rl.sh --algorithm random --replay_buffer_save_folder $replay_buffer_save_folder --buffer_save_path $buffer_save_path $common_args

# if the curiosity module is world_model, we need to call train_world_model.sh as well before. 

#common_args="--algorithm ${ARGS["algorithm"]} --gamma ${ARGS["gamma"]} --similarity_metric ${ARGS["similarity_metric"]} --observation-embedder ${ARGS["observation_embedder"]} --curiosity-module ${ARGS["curiosity_module"]} --max_steps ${ARGS["max_steps"]} --timesteps ${ARGS["timesteps"]} --controller ${ARGS["controller"]}"



# create a function. The function reads set argum