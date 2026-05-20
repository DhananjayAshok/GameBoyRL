# DELTA TODO: 2: Change this to not loop over init state groups but just use init state name as init state group names. Do not gather trajectories here.  
# This becomes the curiosity exploration only vertical for task discovery. 

#!/usr/bin/env bash

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
python scripts/create_task_dictionary.py
source scripts/all_train_states.sh

# Script-specific defaults and required args
declare -A ARGS
REQUIRED_ARGS=()
populate_array CREATE_TRAJ_ESSENTIALS REQUIRED_ARGS
populate_dict CREATE_TRAJ_DEFAULTS ARGS


# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# error out if run_name is not set, is default or is iterative
if [[ -z "${ARGS["run_name"]}" ]] || [[ "${ARGS["run_name"]}" == "default" ]] || [[ "${ARGS["run_name"]}" == "iterative" ]]; then
    echo "Error: --run_name is required and cannot be 'default' or 'iterative'."
    usage
fi

game="${ARGS["game"]}"
IFS=',' read -ra init_states_arr <<< "${TRAIN_STATES[$game]}"
for init_state in "${init_states_arr[@]}"; do
    ARGS["init_state"]=$init_state
    arg_string=$(args_to_flags_subset ARGS CREATE_TRAJ_ARG_KEYS)
    bash scripts/create_traj.sh $arg_string
done