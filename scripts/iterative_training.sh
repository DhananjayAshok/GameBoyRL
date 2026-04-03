#!/usr/bin/env bash

source scripts/utils.sh

# Define Defaults for default_rl.sh
declare -A ARGS
REQUIRED_ARGS=()
populate_dict SWEEP_DEFAULTS ARGS
populate_array SWEEP_ESSENTIALS REQUIRED_ARGS

ARGS["n_agents"]=4
ARGS["sweep"]=false
ARGS["combination_buffer_sweep"]=true
combination_buffer_ocr_alphas=(0.5 1.0)


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


sweeping=false
if [ "${ARGS["sweep"]}" == "true" ]; then
    run_name="iterative_sweep"
    sweeping=true
else
    run_name="iterative_agent"
fi
ocr_alpha_sweep=false
if [ "${ARGS["combination_buffer_sweep"]}" == "true" ] && [ "${ARGS["curiosity_module"]}" == "combinationbuffer" ]; then
    ocr_alpha_sweep=true
fi
if [ "$ocr_alpha_sweep" = true ] && [ "${ARGS["ocr_alpha"]}" != "0.0" ]; then
    ARGS["ocr_alpha"]=0.0
    echo "Warning: combination_buffer_sweep is true but ocr_alpha is not 0.0. Setting ocr_alpha to 0.0 for sweeping."
fi

replay_buffer_save_folder=iterative/${ARGS["init_state"]}/$run_name/
ARGS["latest_replay_buffer_folder"]=$replay_buffer_save_folder
ARGS["replay_buffer_save_folder"]=$replay_buffer_save_folder



prev_buffer_load_path="none"
log_folder="${ARGS["game"]}/iterative/$run_name/${ARGS["init_state"]}/"

function call_agent(){
    local buffer_load_path="$1"
    local buffer_save_path="$2"
    ARGS["buffer_save_path"]=$buffer_save_path
    ARGS["buffer_load_path"]=$buffer_load_path
    if [ "$sweeping" = true ]; then
        argstring=$(args_to_flags_subset ARGS SWEEP_ARG_KEYS)
        bash scripts/sweep.sh $argstring
        if [ "$ocr_alpha_sweep" = true ]; then
            for ocr_alpha in "${combination_buffer_ocr_alphas[@]}"; do
                ARGS["ocr_alpha"]=$ocr_alpha
                argstring=$(args_to_flags_subset ARGS SWEEP_ARG_KEYS)
                bash scripts/sweep.sh $argstring
            done
            ARGS["ocr_alpha"]=0.0 # reset to default after sweep
        fi
    else
        argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)    
        bash scripts/default_rl.sh $argstring
        if [ "$ocr_alpha_sweep" = true ]; then
            for ocr_alpha in "${combination_buffer_ocr_alphas[@]}"; do
                ARGS["ocr_alpha"]=$ocr_alpha
                argstring=$(args_to_flags_subset ARGS TRAINING_ARG_KEYS)    
                bash scripts/default_rl.sh $argstring
            done
            ARGS["ocr_alpha"]=0.0 # reset to default after sweep
        fi
    fi
}

function train_world_model(){
    local buffer_save_path="$1"
    ARGS["buffer_save_path"]=$buffer_save_path
    # Don't bother with buffer_load_path, this will just make the WM train from scratch on all trajectories in the replay buffer folder. 
    argstring=$(args_to_flags_subset ARGS WORLD_MODEL_ARG_KEYS)
    bash scripts/train_world_model.sh $argstring 
}

## Execution starts here

if [ "$sweeping" = true ]; then
    all_buffer_save_paths=iterative/${ARGS["init_state"]}/sweep_
else
    all_buffer_save_paths=iterative/${ARGS["init_state"]}/${ARGS["algorithm"]}_agent_
fi


buffer_save_path=${all_buffer_save_paths}0
prev_buffer_save_path=$buffer_save_path

call_agent "none" $buffer_save_path

if [ "${ARGS["curiosity_module"]}" == "world_model" ]; then
    train_world_model $buffer_save_path    
fi

prev_buffer_load_path=$prev_buffer_save_path
buffer_save_path=${all_buffer_save_paths}1

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
    buffer_save_path="${all_buffer_save_paths}$((i+2))"
done

# for now do grouping of trajectories here:
cd cleanrl
observation_embedder_part=""
if [[ "${ARGS["observation_embedder"]}" != "none" ]]; then
    observation_embedder_part="--observation_embedder ${ARGS["observation_embedder"]}"
fi
if [[ "${ARGS["embedder_load_path"]}" != "none" ]]; then
    observation_embedder_part+=" --embedder_load_path $storage_dir/${ARGS["observation_embedder"]}/${ARGS["game"]}/${ARGS["embedder_load_path"]} "
fi

echo python cleanrl_utils/group_trajectories.py --replay_buffer_folder $storage_dir/replay_buffers/${ARGS["game"]}/$replay_buffer_save_folder --save_path $storage_dir/grouped_trajectories/${ARGS["game"]}/$replay_buffer_save_folder/ $observation_embedder_part
python cleanrl_utils/group_trajectories.py --replay_buffer_folder $storage_dir/replay_buffers/${ARGS["game"]}/$replay_buffer_save_folder --save_path $storage_dir/grouped_trajectories/${ARGS["game"]}/$replay_buffer_save_folder/ $observation_embedder_part