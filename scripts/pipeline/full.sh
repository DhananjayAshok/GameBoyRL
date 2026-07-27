#!/usr/bin/env bash
# End-to-end driver: runs the zeroshot-with-curiosity proposal/practice pipeline for
# a game, then fine-tunes the VLM on the train/validation datasets it produces.
# Assumes a VLM server is already serving --model_name (does NOT start one) — bring up
# the server (e.g. scripts/core/serve_vllm.sh, or any `vllm serve` equivalent)
# before calling this.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array VLM_ESSENTIALS REQUIRED_ARGS   # game, model_name, vlm_kind
REQUIRED_ARGS+=("run_name")

ARGS["executor"]="history"
ARGS["push_to_hub"]=true
ARGS["batch_size"]=4
ARGS["mode"]="both"

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

game="${ARGS["game"]}"
mode="${ARGS["mode"]}"
executor="${ARGS["executor"]}"
model_save_name="${ARGS["model_name"]##*/}"
proposed_dir="$storage_dir/proposed_tasks/${game}/${model_save_name}"

# Stage 1: run the data-collection legs for this mode, and resolve dataset_dir — the
# directory holding train_dataset.csv / validation_dataset.csv. In two modes that is a
# practice dir; in "both" it is the merged output. Every path is re-derived here rather
# than passed back up, matching how the rest of the pipeline works.
case "$mode" in
    curiosity_only)
        curiosity_flags=$(args_to_flags_subset ARGS CURIOSITY_TASKS_ARG_KEYS)
        bash scripts/pipeline/curiosity_all_tasks.sh $curiosity_flags --do_guidance_and_practice true || exit 1
        dataset_dir="$proposed_dir/curiosity/${ARGS["run_name"]}/practice_${executor}"
        ;;
    zeroshot_only)
        propose_flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
        bash scripts/pipeline/propose_and_attempt_all.sh $propose_flags \
            --extra none --propose_only false --do_guidance_and_practice true || exit 1
        dataset_dir="$proposed_dir/zeroshot/zeroshot_tasks_${executor}_attempts/practice_${executor}"
        ;;
    both)
        propose_flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
        bash scripts/pipeline/curiosity_and_zeroshot_all.sh $propose_flags || exit 1
        dataset_dir=$(merged_dataset_dir "$game" "$model_save_name" "${ARGS["run_name"]}")
        ;;
    *)
        echo "Error: unknown --mode '$mode'. Choose one of: curiosity_only, zeroshot_only, both."
        exit 1 ;;
esac

# Stage 2: fine-tune on whichever dataset this mode produced.
train_file="$dataset_dir/train_dataset.csv"
validation_file="$dataset_dir/validation_dataset.csv"

echo "Mode '$mode' -> dataset_dir: $dataset_dir"

if [[ ! -f "$train_file" ]]; then
    echo "Error: train dataset not found at $train_file. The mode '$mode' data-collection stage did not produce it."
    exit 1
fi

# If the VLM server is still running, stop it before fine-tuning so it isn't holding VRAM.
# Failing here is fine (it just means nothing was running).
# NOTE: stop_vllm.sh routes through ~/vllm_scripts if present, else SIGTERMs the vllm
# process group its generic serve path recorded at launch.
bash scripts/core/stop_vllm.sh

# The mode is part of the run name, and therefore of the checkpoint dir, the hub repo, the
# served model name and the benchmark CSV. Without it, running two modes with the same
# --run_name silently overwrites the first one's checkpoint AND its benchmark results.
# serve_and_benchmark.sh re-derives this same string from --game/--run_name/--mode.
# Fine-tuning is currently echoed rather than run, so the data-collection legs can be
# exercised end-to-end without spending a training job. Restore by dropping the echo.
bash scripts/vlm/train_vlm.sh \
    --train_file "$train_file" \
    --validation_file "$validation_file" \
    --model_name "${ARGS["model_name"]}" \
    --run_name "${game}-${ARGS["run_name"]}-${mode}" \
    --push_to_hub "${ARGS["push_to_hub"]}" \
    --batch_size "${ARGS["batch_size"]}" || exit 1
