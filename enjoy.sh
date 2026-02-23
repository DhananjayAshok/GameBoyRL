#!/usr/bin/env bash

source configs/config.env || { echo "configs/config.env not found"; exit 1; }
source setup/.venv/bin/activate || { echo "Virtual environment not found."; exit 1; }

# Define Defaults
declare -A ARGS
ARGS["max_steps"]=200
ARGS["similarity_metric"]="cosine"
ARGS["observation_embedder"]="random_patch"
ARGS["curiosity_module"]="embedbuffer"
ARGS["buffer_save_path"]=""
ARGS["buffer_load_path"]=""

# Define Required Keys
REQUIRED_ARGS=("algorithm" "exp_name" "game" "controller" "env" "init_state")

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
echo "Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# Logic here:
exp_name="${ARGS["exp_name"]}"
test_env_id="poke_worlds-${ARGS["game"]}-${ARGS["env"]}-${ARGS["init_state"]}-${ARGS["controller"]}-${ARGS["max_steps"]}-True"
model_save_path="$storage_dir/models/$exp_name/"
buffer_arg_part=""
if [[ -n "${ARGS["buffer_save_path"]}" ]]; then
    buffer_arg_part+="--buffer_save_path $storage_dir/${ARGS["curiosity_module"]}/${ARGS["buffer_save_path"]} "
fi

if [[ -n "${ARGS["buffer_load_path"]}" ]]; then
    buffer_arg_part+="--buffer_load_path $storage_dir/${ARGS["curiosity_module"]}/${ARGS["buffer_load_path"]} "
fi


cd cleanrl

python cleanrl_utils/enjoy.py --exp-name ${ARGS["algorithm"]}_curiosity --model_path $model_save_path/model.pt \
    --env-id $test_env_id --save-name $exp_name \
    --similarity_metric ${ARGS["similarity_metric"]} --observation_embedder ${ARGS["observation_embedder"]} \
    --curiosity_module ${ARGS["curiosity_module"]} $buffer_arg_part

cd ..