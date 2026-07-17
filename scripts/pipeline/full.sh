#!/usr/bin/env bash
# End-to-end driver: runs the zeroshot-with-curiosity proposal/practice pipeline for
# a game, then fine-tunes the VLM on the train/validation datasets it produces.
# Assumes a VLM server is already serving --model_name (does NOT start one) — bring up
# the server (e.g. vllm_scripts/serve_vllm_*.sh) before calling this.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array VLM_ESSENTIALS REQUIRED_ARGS   # game, model_name, vlm_kind
REQUIRED_ARGS+=("run_name")

ARGS["executor"]="history"
ARGS["push_to_hub"]=true
ARGS["batch_size"]=4

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

# Stage 1: propose zeroshot-with-curiosity tasks and run guidance/practice/clean/
# create_dataset for every init_state. Only the 5 shared keys are forwarded; the
# subscript fills the rest (incl. do_guidance_and_practice=true) from its defaults.
propose_flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
bash scripts/pipeline/propose_zeroshot_with_curiosity_all.sh $propose_flags || exit 1

bash scripts/core/stop_vllm.sh # If the VLM server is still running, stop it before fine-tuning. If we error out here its fine. 

# Stage 2: fine-tune on the datasets create_dataset.py wrote into the practice dir.
# Path mirrors what guidance_and_practice derives:
#   dirname(success_trajectories)/practice_{executor}
# under the zeroshot_with_curiosity attempts dir (see propose_and_attempt_all.sh).
model_save_name="${ARGS["model_name"]##*/}"
practice_dir="$storage_dir/proposed_tasks/${ARGS["game"]}/${model_save_name}/zeroshot/zeroshot_tasks_prior_zeroshot_with_curiosity_${ARGS["executor"]}_attempts/practice_${ARGS["executor"]}"
train_file="$practice_dir/train_dataset.csv"
validation_file="$practice_dir/validation_dataset.csv"

if [[ ! -f "$train_file" ]]; then
    echo "Error: train dataset not found at $train_file. The proposal/practice stage did not produce it."
    exit 1
fi

bash scripts/vlm/train_vlm.sh \
    --train_file "$train_file" \
    --validation_file "$validation_file" \
    --model_name "${ARGS["model_name"]}" \
    --run_name "${ARGS["game"]}-${ARGS["run_name"]}" \
    --push_to_hub "${ARGS["push_to_hub"]}" \
    --batch_size "${ARGS["batch_size"]}" || exit 1
