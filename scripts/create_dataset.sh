
#!/usr/bin/env bash

source scripts/utils.sh

declare -A ARGS
REQUIRED_ARGS=()


populate_dict CREATE_DATASET_DEFAULTS ARGS
populate_array  CREATE_DATASET_ESSENTIALS REQUIRED_ARGS


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

model_name="${ARGS["model_name"]}"
vlm_kind="${ARGS["vlm_kind"]}"
game="${ARGS["game"]}"
max_new_tokens="${ARGS["max_new_tokens"]}"
max_trajectories_per_group="${ARGS["max_trajectories_per_group"]}"
lookback="${ARGS["lookback"]}"
describe_pairs="${ARGS["describe_pairs"]}"
safety_rollback="${ARGS["safety_rollback"]}"
init_state_group="${ARGS["init_state_group"]}"
run_name="${ARGS["run_name"]}"
overwrite="${ARGS["overwrite"]}"
verbose="${ARGS["verbose"]}"
push_to_hub="${ARGS["push_to_hub"]}"

# if describe_paris is true, t, yes or y then 
if [[ "$describe_pairs" == "true" || "$describe_pairs" == "yes" || "$describe_pairs" == "y" ]]; then
    describe_pairs_flag="--describe_pairs"
else
    describe_pairs_flag=""
fi

if [[ "$overwrite" == "true" || "$overwrite" == "yes" || "$overwrite" == "y" ]]; then
    overwrite_flag="--overwrite"
else
    overwrite_flag=""
fi

if [[ "$verbose" == "true" || "$verbose" == "yes" || "$verbose" == "y" ]]; then
    verbose_flag="--verbose"
else
    verbose_flag=""
fi

if [[ "$push_to_hub" == "true" || "$push_to_hub" == "yes" || "$push_to_hub" == "y" ]]; then
    push_to_hub_flag="--push_to_hub"
else
    push_to_hub_flag=""
fi

trajectory_path=$storage_dir/grouped_trajectories/${game}/${run_name}/${init_state_group}/grouped_global_high_reward_trajectories.pkl

common_call=" --trajectory_path $trajectory_path --model_name $model_name --vlm_kind $vlm_kind --game $game --max_new_tokens $max_new_tokens $overwrite_flag $verbose_flag" 

echo "Running Sparse Annotation:"
python create_dataset.py $common_call infer --max_trajectories_per_group "$max_trajectories_per_group" --lookback "$lookback" $describe_pairs_flag 

echo "Running Reasoning Annotation:"
python create_dataset.py $common_call reason --safety_rollback "$safety_rollback"

echo "Running Collation:"
python create_dataset.py $common_call save $push_to_hub_flag