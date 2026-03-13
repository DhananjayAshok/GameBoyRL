#!/usr/bin/env bash

source scripts/utils.sh

# Define Defaults
declare -A ARGS
ARGS["test_env"]="none"
ARGS["test_init_state"]="none"
ARGS["train_only"]="false"
ARGS["eval_only"]="false"


# Define Required Keys
REQUIRED_ARGS=()


populate_dict TRAINING_DEFAULTS ARGS
populate_array TRAINING_ESSENTIALS REQUIRED_ARGS

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
argstring=() 
cd cleanrl

# Logic here:


train_env_id=$(get_string_from_args "env_id" ARGS)
if [[ -z "$train_env_id" ]]; then
    echo "Error: Failed to get train_env_id"
    exit 1
fi
train_env_id=$train_env_id-False

exp_name=$(get_string_from_args "exp_name" ARGS)
if [[ -z "$exp_name" ]]; then
    echo "Error: Failed to generate exp_name"
    exit 1
fi

# check eval only flag, if true then skip training and go to evaluation
if [[ "${ARGS["eval_only"]}" == "true" || "${ARGS["eval_only"]}" == "yes" || "${ARGS["eval_only"]}" == "y" ]]; then
    echo "Eval only flag is set. Skipping training and going to evaluation."
else
    arg_string=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
    bash scripts/train.sh $arg_string
fi

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


ARGS["exp_name"]=$exp_name
arg_string=$(args_to_flags_subset ARGS EVALUATION_ARG_KEYS)
bash scripts/enjoy.sh $arg_string