#!/usr/bin/env bash
# Trains an observation encoder on replay buffer data collected for a given game,
# saving the resulting embedding model to storage. The encoder is used as a
# reusable similarity-based feature extractor for curiosity modules in subsequent
# RL training runs (embedder_load_path argument).

source scripts/core/utils.sh

# Define Defaults
declare -A ARGS
REQUIRED_ARGS=()
populate_dict ALL_DEFAULTS ARGS
populate_array ESSENTIAL_ARGS REQUIRED_ARGS

ARGS["subfolder"]="none"

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


cd cleanrl

# Logic here:

subfolder=""
if [[ "${ARGS["subfolder"]}" != "none" ]]; then
    subfolder="${ARGS["subfolder"]}"
fi

replay_buffer_folder=$storage_dir/replay_buffers/${ARGS["game"]}/${subfolder}/
save_path=$(path_of observation_embedder_dir --game "${ARGS["game"]}" --run_name global)


echo "Training Observation Embedder Model:"

python cleanrl_utils/train_observation_encoder.py --seed 1 --replay_buffer_folder $replay_buffer_folder \
    --track --wandb-project-name $WANDB_PROJECT \
    --save_path $save_path

cd ..