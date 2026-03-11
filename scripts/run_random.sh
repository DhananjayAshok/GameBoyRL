#!/usr/bin/env bash

source scripts/utils.sh

# Define Defaults for default_rl.sh
declare -A ARGS
ARGS["similarity_metric"]="cosine"
ARGS["observation_embedder"]="random_patch"
ARGS["embedder_load_path"]="none"
ARGS["curiosity_module"]="embedbuffer"
ARGS["max_steps"]=50
ARGS["timesteps"]=1000000
ARGS["controller"]="low_level"
# Script specific defaults

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
buffer_save_path=${ARGS["init_state"]}/random/
prev_buffer_save_path=$buffer_save_path

# First, run a random agent to populate the curiosity module buffer
log_folder=$storage_dir/logs/${ARGS["game"]}/random/

bash scripts/default_rl.sh --algorithm random --buffer_save_path $buffer_save_path --similarity_metric ${ARGS["similarity_metric"]} --observation_embedder ${ARGS["observation_embedder"]} --embedder_load_path ${ARGS["observation_embedder"]} --curiosity_module ${ARGS["curiosity_module"]} --max_steps ${ARGS["max_steps"]} --timesteps ${ARGS["timesteps"]} --controller ${ARGS["controller"]} --replay_buffer_save_folder $replay_buffer_save_folder --game ${ARGS["game"]} --env default --init_state ${ARGS["init_state"]} --log_folder $log_folder