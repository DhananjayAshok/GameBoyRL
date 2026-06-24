#!/usr/bin/env bash
# Batch version of propose_zeroshot_with_curiosity: checks that infer_tasks has
# been run, then delegates to propose_and_attempt_all twice — once with extra=none
# and once with extra=zeroshot_with_curiosity.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
python scripts/python/create_task_dictionary.py || { echo "Could not regenerate train states"; exit 1; }
source scripts/core/all_train_states.sh

declare -A ARGS
REQUIRED_ARGS=()

populate_array VLM_ESSENTIALS REQUIRED_ARGS
populate_dict PROPOSE_AND_ATTEMPT_DEFAULTS ARGS
unset ARGS["extra"]

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

model_save_name="${ARGS["model_name"]##*/}"
curiosity_dir="$storage_dir/proposed_tasks/${ARGS["game"]}/${model_save_name}/curiosity/${ARGS["run_name"]}"

user_do_guidance="${ARGS["do_guidance_and_practice"]}"

# Ensure the curiosity prior exists. curiosity_all_tasks runs create_traj (skips if
# grouped trajectories already exist) + infer_tasks (skips if annotation exists) for
# every init_state. overwrite=false so existing curiosity output is reused, never
# regenerated; do_guidance_and_practice=false since we only need the annotation prior.
saved_overwrite="${ARGS["overwrite"]}"
ARGS["overwrite"]="false"
ARGS["do_guidance_and_practice"]="false"
curiosity_flags=$(args_to_flags_subset ARGS CURIOSITY_TASKS_ARG_KEYS)
bash scripts/pipeline/curiosity_all_tasks.sh $curiosity_flags || exit 1
ARGS["overwrite"]="$saved_overwrite"

if [[ ! -f "$curiosity_dir/trajectory_annotation.json" || ! -f "$curiosity_dir/trajectory_annotation.pkl" ]]; then
    echo "Error: curiosity annotation still missing at $curiosity_dir after curiosity_all_tasks. Aborting."
    exit 1
fi

# Stage 1: zeroshot baseline — only proposal needed as prior for stage 2.
ARGS["extra"]="none"
ARGS["propose_only"]="true"
ARGS["do_guidance_and_practice"]="false"
flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
bash scripts/pipeline/propose_and_attempt_all.sh $flags || exit 1

# Stage 2: the meaningful output — use zeroshot_tasks_prior_zeroshot_with_curiosity_attempts/success_trajectories for guidance_and_practice.
ARGS["extra"]="zeroshot_with_curiosity"
ARGS["propose_only"]="false"
ARGS["do_guidance_and_practice"]="$user_do_guidance"
flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
bash scripts/pipeline/propose_and_attempt_all.sh $flags || exit 1
