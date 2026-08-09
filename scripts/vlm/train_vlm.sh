#!/usr/bin/env bash
# Fine-tunes a VLM using LoRA via the llm-utils training CLI (train.py --modality vlm)
# on a provided dataset file. Saves the LoRA adapter to storage under the run name.
# Supports optional push to HuggingFace Hub via --push_to_hub. Uses early stopping
# and best-model checkpointing with a fixed 85/15 train/validation split.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

# Define defaults and required args.
# These should be specific to this script and not shared across scripts (that is handled below).
declare -A ARGS
ARGS["batch_size"]="8"
ARGS["num_train_epochs"]="2"
ARGS["lora_rank"]="64"
ARGS["lora_alpha"]="128"
ARGS["learning_rate"]="2e-4"
ARGS["weight_decay"]="0.01"
ARGS["overwrite"]=false
ARGS["push_to_hub"]=false
ARGS["validation_file"]=none
ARGS["action_loss_weight"]=0.25

REQUIRED_ARGS=("train_file" "model_name" "run_name")

# Handle parsing and input errors below:
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
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"
            exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done

function usage() {
    echo "$USAGE_STR"
    exit 1
}

# Parser
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
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

# Validation
for req in "${REQUIRED_ARGS[@]}"; do
    echo $req : "${ARGS[$req]}"
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




model_name="${ARGS["model_name"]}"
# '##*/' (after the LAST slash), matching model_name.split("/")[-1] in Python. '#*/' strips
# only up to the FIRST slash, which agrees for a one-slash name like google/gemma-4-31b-it
# but not for org/team/model — and the served-model name is rebuilt on the Python side.
model_save_name="${model_name##*/}"

# if overwrite is true, t, yes or y, set --restore_from_checkpoint False, else set it to empty string
if [[ "${ARGS["overwrite"]}" == "true" || "${ARGS["overwrite"]}" == "yes" || "${ARGS["overwrite"]}" == "y" ]]; then
    overwrite_flag="--resume_from_checkpoint False --overwrite_final True"
else
    overwrite_flag=""
fi

# if push_to_hub is true, t, yes or y, set --push_to_hub True, else set it to empty string
if [[ "${ARGS["push_to_hub"]}" == "true" || "${ARGS["push_to_hub"]}" == "yes" || "${ARGS["push_to_hub"]}" == "y" ]]; then
    push_to_hub_flag="--push_to_hub True --hub_model_id ${huggingface_repo_namespace}/${ARGS["run_name"]}-$model_save_name"
else
    push_to_hub_flag=""
fi

# If a validation file is provided, forward it and let the trainer use it directly
# (a leakage-free split is the caller's responsibility). Otherwise fall back to an internal
# 0.85 train/validation split of the train file.
if [[ "${ARGS["validation_file"]}" != "none" ]]; then
    validation_flag="--validation_file ${ARGS["validation_file"]}"
else
    validation_flag="--train_validation_split 0.85"
fi

# If an action_loss_weight is provided, upweight the "Action:" tokens in the completion
# loss (see llm-utils train.py --action_loss_weight). Otherwise weight all tokens equally.
if [[ "${ARGS["action_loss_weight"]}" != "none" ]]; then
    action_loss_flag="--action_loss_weight ${ARGS["action_loss_weight"]}"
else
    action_loss_flag=""
fi




bash scripts/core/llm-utils.sh python train.py --training_kind sft --modality vlm --model_name ${ARGS["model_name"]} \
        --output_dir $storage_dir/models/${ARGS["run_name"]}/$model_save_name \
        --train_file ${ARGS["train_file"]}  \
        --run_name vlm-sft-${ARGS["run_name"]}-$model_save_name \
        --per_device_train_batch_size ${ARGS["batch_size"]} --per_device_eval_batch_size ${ARGS["batch_size"]} \
        $validation_flag \
        --logging_strategy steps --logging_steps 200 \
        --save_strategy epoch --save_steps 0.5 \
        --eval_strategy epoch --eval_steps 0.5 \
        --early_stopping_patience 2 --load_best_model_at_end \
        --num_train_epochs ${ARGS["num_train_epochs"]} \
        --lora_r ${ARGS["lora_rank"]} \
        --lora_alpha ${ARGS["lora_alpha"]} \
        --learning_rate ${ARGS["learning_rate"]} \
        --weight_decay ${ARGS["weight_decay"]} $action_loss_flag $overwrite_flag $push_to_hub_flag || exit 1
