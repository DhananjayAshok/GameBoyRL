#!/usr/bin/env bash

source scripts/utils.sh

# Define Defaults
declare -A ARGS
ARGS["max_steps"]=200
ARGS["controller"]="low_level"
ARGS["similarity_metric"]="cosine"
ARGS["observation_embedder"]="random_patch"
ARGS["embedder_load_path"]="none"
ARGS["curiosity_module"]="embedbuffer"
ARGS["buffer_save_path"]="none"
ARGS["buffer_load_path"]="none"
ARGS["model_dir"]="none"

# Define Required Keys
REQUIRED_ARGS=("algorithm" "exp_name" "game" "env" "init_state")

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
                echo "Error: Unknown flag --$FLAG. Allowed flags are: ${ALLOWED_FLAGS[*]}"
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
echo "Script $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# Logic here:
exp_name="${ARGS["exp_name"]}"
test_env_id=$(get_env_id --game "${ARGS["game"]}" --env "${ARGS["env"]}" --init_state "${ARGS["init_state"]}" \
    --controller "${ARGS["controller"]}" --max_steps "${ARGS["max_steps"]}")
if [[ -z "$test_env_id" ]]; then
    echo "Error: Failed to construct test environment ID. Please check your input parameters."
    exit 1
fi
test_env_id="${test_env_id}-True"

if [[ "${ARGS["model_dir"]}" != "none" ]]; then
    model_save_path="$storage_dir/models/${ARGS["model_dir"]}/$exp_name/"
else
    model_save_path="$storage_dir/models/$exp_name/"
fi

extra_arg_part=""
if [[ "${ARGS["buffer_save_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_save_path $storage_dir/curiosity_buffers/${ARGS["curiosity_module"]}/${ARGS["game"]}/${ARGS["buffer_save_path"]} "
fi

if [[ "${ARGS["buffer_load_path"]}" != "none" ]]; then
    extra_arg_part+="--buffer_load_path $storage_dir/curiosity_buffers/${ARGS["curiosity_module"]}/${ARGS["game"]}/${ARGS["buffer_load_path"]} "
fi
if [[ "${ARGS["embedder_load_path"]}" != "none" ]]; then
    extra_arg_part+="--embedder_load_path $storage_dir/${ARGS["observation_embedder"]}/${ARGS["game"]}/${ARGS["embedder_load_path"]} "
fi


cd cleanrl

python cleanrl_utils/enjoy.py --exp-name ${ARGS["algorithm"]}_curiosity --model_path $model_save_path \
    --env-id $test_env_id --save-name $exp_name \
    --similarity_metric ${ARGS["similarity_metric"]} --observation_embedder ${ARGS["observation_embedder"]} \
    --curiosity_module ${ARGS["curiosity_module"]} $extra_arg_part

cd ..