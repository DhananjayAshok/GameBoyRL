#!/usr/bin/env bash
# Two-stage proposal for a single init_state: first proposes with no prior
# (extra=none), then proposes with extra=zeroshot_with_curiosity, each followed
# by an attempt run. Checks that infer_tasks has been run before starting.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array PROPOSE_AND_ATTEMPT_ESSENTIALS REQUIRED_ARGS
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

if [[ ! -f "$curiosity_dir/trajectory_annotation.json" ]]; then
    echo "Error: trajectory_annotation.json not found at $curiosity_dir. Run infer_tasks first."
    exit 1
fi
if [[ ! -f "$curiosity_dir/trajectory_annotation.pkl" ]]; then
    echo "Error: trajectory_annotation.pkl not found at $curiosity_dir. Run infer_tasks first."
    exit 1
fi

user_do_guidance="${ARGS["do_guidance_and_practice"]}"

# Stage 1: zeroshot baseline — only proposal needed as prior for stage 2.
ARGS["extra"]="none"
ARGS["propose_only"]="true"
ARGS["do_guidance_and_practice"]="false"
flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
bash scripts/pipeline/propose_and_attempt.sh $flags || exit 1

# Stage 2: the meaningful output — use zeroshot_tasks_prior_zeroshot_with_curiosity_attempts/success_trajectories for guidance_and_practice.
ARGS["extra"]="zeroshot_with_curiosity"
ARGS["propose_only"]="false"
ARGS["do_guidance_and_practice"]="$user_do_guidance"
flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
bash scripts/pipeline/propose_and_attempt.sh $flags || exit 1
