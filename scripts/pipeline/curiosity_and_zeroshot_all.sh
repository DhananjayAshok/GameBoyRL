#!/usr/bin/env bash
# Runs both data-collection verticals for a game and merges their datasets into one
# fine-tuning set:
#
#   leg 1  curiosity — RL exploration -> infer_tasks -> guidance -> practice -> dataset
#   leg 2  zeroshot  — propose (extra=none) -> attempt -> guidance -> practice -> dataset
#   merge  create_dataset.py merge_practices, tagging each row with its source leg
#
# The curiosity annotation is NOT used as a proposal prior. That arm (--extra
# zeroshot_with_curiosity) measured near-inert — only ~3% of its proposed task strings
# carried over verbatim from the prior, because get_extra_context samples 20 strings from a
# pool spanning every init_state and the prompt tells the model not to use them blindly. The
# curiosity vertical now contributes by producing training data directly, which is both
# cheaper (its trajectories already exist from RL) and separately measurable.
#
# Called by full.sh --mode both. --extra zeroshot_with_curiosity remains available in
# propose_and_attempt_all.sh for anyone who wants that arm manually.

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

# Leg 1 — curiosity. curiosity_all_tasks runs create_traj (skips if grouped trajectories
# already exist) + infer_tasks (skips if the annotation exists) for every init_state, then
# guidance/practice/clean/create_dataset. overwrite=false so existing curiosity output is
# reused rather than regenerated — that is also what makes the practice stage resumable,
# since practice_tasks returns early when results.csv exists.
# The curiosity leg follows the caller's do_guidance_and_practice, same as the zeroshot legs:
# with it on, all three verticals produce training data and are merged below; with it off,
# this script only produces proposals and the annotation prior.
saved_overwrite="${ARGS["overwrite"]}"
ARGS["overwrite"]="false"
ARGS["do_guidance_and_practice"]="$user_do_guidance"
curiosity_flags=$(args_to_flags_subset ARGS CURIOSITY_TASKS_ARG_KEYS)
bash scripts/pipeline/curiosity_all_tasks.sh $curiosity_flags || exit 1
ARGS["overwrite"]="$saved_overwrite"

if [[ ! -f "$curiosity_dir/trajectory_annotation.json" || ! -f "$curiosity_dir/trajectory_annotation.pkl" ]]; then
    echo "Error: curiosity annotation still missing at $curiosity_dir after curiosity_all_tasks. Aborting."
    exit 1
fi

# Leg 2 — zeroshot baseline, now a full leg (propose + attempt + practice) rather than a
# proposal that only fed leg 3's prior. Attempting these tasks is what makes the prior's
# value measurable: without it there is no baseline arm to compare leg 3 against.
ARGS["extra"]="none"
ARGS["propose_only"]="false"
flags=$(args_to_flags_subset ARGS PROPOSE_AND_ATTEMPT_ARG_KEYS)
bash scripts/pipeline/propose_and_attempt_all.sh $flags || exit 1

# --- Merge the two legs' datasets -------------------------------------------------------
# Each leg's practice dir is derived the same way its own stage derives it:
#   curiosity  dirname(trajectory_annotation)/practice_<executor>
#   zeroshot   <tasks jsonl minus .jsonl>_<executor>_attempts/practice_<executor>
executor="${ARGS["executor"]}"
zeroshot_base="$storage_dir/proposed_tasks/${ARGS["game"]}/${model_save_name}/zeroshot"
curiosity_practice="$curiosity_dir/practice_${executor}"
zeroshot_practice="$zeroshot_base/zeroshot_tasks_${executor}_attempts/practice_${executor}"

echo "Practice dirs:"
echo "  curiosity: $curiosity_practice"
echo "  zeroshot:  $zeroshot_practice"

if [[ "$user_do_guidance" != "true" ]]; then
    echo "do_guidance_and_practice=$user_do_guidance: the legs produced no datasets, so there is nothing to merge. Skipping."
    exit 0
fi

merged_dir=$(merged_dataset_dir "${ARGS["game"]}" "$model_save_name" "${ARGS["run_name"]}")
echo "Merging the two legs into $merged_dir"
bash scripts/vlm/merge_practices.sh \
    --practice_paths "$curiosity_practice,$zeroshot_practice" \
    --save_path "$merged_dir" || exit 1
