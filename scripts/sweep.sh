#!/usr/bin/env bash

source scripts/utils.sh

# Define Defaults for default_rl.sh
declare -A ARGS
REQUIRED_ARGS=()
populate_array SWEEP_ESSENTIALS REQUIRED_ARGS
populate_dict SWEEP_DEFAULTS ARGS



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
if [[ "${ARGS["model_dir"]}" != "none" ]]; then
    model_save_path="$storage_dir/models/${ARGS["model_dir"]}/"
    rm -rf model_save_path/* # clear the model save path to ensure we don't have old models lying around.
else
    echo "Sweep requires you to specify a --model_dir"
    exit 1
fi

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done


SEEDS=(0 1 2 3 4)
GAMMAS=(0.99 0.995 0.999)
ALGORITHMS=(dqn ppo sac)


true_buffer_save_path=${ARGS["buffer_save_path"]}
ARGS["buffer_save_path"]="none" # the first eval of the sweep is just to assess test reward, not to save to buffers. 
for seed in "${SEEDS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for algorithm in "${ALGORITHMS[@]}"; do
            ARGS["seed"]=$seed
            ARGS["gamma"]=$gamma
            ARGS["algorithm"]=$algorithm
            argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
            bash scripts/default_rl.sh $argstring
        done
    done
done
ARGS["buffer_save_path"]="$true_buffer_save_path"

# Keep only the best
arg_str=""
if [[ "${ARGS["replay_buffer_save_folder"]}" != "none" ]]; then
    arg_str+="--replay_buffer_save_folder $storage_dir/replay_buffers/${ARGS["game"]}/${ARGS["replay_buffer_save_folder"]} "
else
    ARGS["clear_loser_replay_buffer"]="false" # force false if replay buffer isn't saved in the first place. 
fi


arg_str+="--best_k ${ARGS["best_k"]} --model_dir $model_save_path"
if [[ "${ARGS["clear_loser_replay_buffer"]}" == "true" ]]; then
    arg_str+=" --clear_loser_replay_buffer"
fi


cd cleanrl
python cleanrl_utils/keep_only_best_models.py $arg_str
cd ..


for seed in "${SEEDS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for algorithm in "${ALGORITHMS[@]}"; do
            ARGS["seed"]=$seed
            ARGS["gamma"]=$gamma
            ARGS["algorithm"]=$algorithm
            argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)
            bash scripts/default_rl.sh $argstring --eval_only true
        done
    done
done