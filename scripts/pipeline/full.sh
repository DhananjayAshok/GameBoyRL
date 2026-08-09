#!/usr/bin/env bash
# End-to-end driver: runs the data-collection legs a --mode selects for one game
# (proposal/attempt and/or curiosity). Each leg's output is a trajectory stem —
# <stem>.json + <stem>.pkl — which scripts/vlm/build_info.sh consumes.
# Assumes a VLM server is already serving --model_name (does NOT start one) — bring up
# the server (e.g. scripts/core/serve_vllm.sh, or any `vllm serve` equivalent)
# before calling this.
#
# Fine-tuning is NOT wired in here: the practice/clean/create_dataset stages that used to
# build train_dataset.csv have been removed from the repo. scripts/vlm/train_vlm.sh still
# exists and takes --train_file/--validation_file, so point it at a dataset yourself.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array VLM_ESSENTIALS REQUIRED_ARGS   # game, model_name, vlm_kind
REQUIRED_ARGS+=("run_name")

ARGS["executor"]="history"
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

# Run the data-collection legs for this mode, and report each leg's trajectory stem.
# Every path is re-derived here rather than passed back up, matching how the rest of the
# pipeline works.
case "$mode" in
    curiosity_only)
        curiosity_flags=$(args_to_flags_subset ARGS CURIOSITY_TASKS_ARG_KEYS)
        bash scripts/pipeline/curiosity_all_tasks.sh $curiosity_flags || exit 1
        stems=("$proposed_dir/curiosity/${ARGS["run_name"]}/trajectory_annotation")
        ;;
    zeroshot_only)
        propose_flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
        bash scripts/pipeline/propose_and_attempt_all.sh $propose_flags \
            --propose_only false || exit 1
        stems=("$proposed_dir/zeroshot/zeroshot_tasks_${executor}_attempts/success_trajectories")
        ;;
    both)
        propose_flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
        bash scripts/pipeline/curiosity_and_zeroshot_all.sh $propose_flags || exit 1
        stems=(
            "$proposed_dir/curiosity/${ARGS["run_name"]}/trajectory_annotation"
            "$proposed_dir/zeroshot/zeroshot_tasks_${executor}_attempts/success_trajectories"
        )
        ;;
    *)
        echo "Error: unknown --mode '$mode'. Choose one of: curiosity_only, zeroshot_only, both."
        exit 1 ;;
esac

# Both halves of the stem must exist for build_info.sh to consume it.
echo "Mode '$mode' produced trajectory stems:"
for stem in "${stems[@]}"; do
    if [[ -f "$stem.json" && -f "$stem.pkl" ]]; then
        echo "  [x] $stem"
    else
        echo "  [ ] $stem  (missing .json and/or .pkl)"
    fi
done
