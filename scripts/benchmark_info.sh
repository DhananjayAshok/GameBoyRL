#!/usr/bin/env bash
# Runs the benchmark suite with info-document hints (the context-engineering arm).
# A sibling of benchmark.sh: same game/executor/model args, but each task goes through an
# InfoHintSupervisor that writes one hint before the executor runs.
#
#   --hint_mode retrieval   selects knowledge by asking, per entry, whether it fits this task
#                           and this screen. Needs --info_docs (from scripts/vlm/build_info.sh).
#   --hint_mode init_state  skips the document and keys off the episode's init state. Needs
#                           --insights_paths, which stage A alone produces.
#
# The flag is --hint_mode, not --mode: across the pipeline scripts --mode always means the
# source selection (curiosity_only / zeroshot_only / both), and the two are orthogonal.
#
# Both options take comma-separated paths, so several sources can be unioned in one run.
# Results land in results/benchmark/<game>/info_<mode>_<executor>_<model>.csv.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args
declare -A ARGS
ARGS["executor"]="simple"
ARGS["executor_vlm_model"]="Qwen/Qwen3-VL-8B-Instruct"   # use "none" for absent optionals, never ""
ARGS["executor_vlm_kind"]="huggingface"   # use "none" for absent optionals, never ""
ARGS["hint_mode"]="retrieval"
ARGS["info_docs"]=none
ARGS["insights_paths"]=none
ARGS["hint_vlm_model"]=none
ARGS["hint_vlm_kind"]=none
ARGS["max_concurrency"]="8"
ARGS["max_steps"]="50"
ARGS["max_resets"]="1"
ARGS["random_sample"]="1"
ARGS["regenerate"]="false"

REQUIRED_ARGS=("game")


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

# Fail here rather than after the emulator has spun up: the required path option depends on
# --hint_mode, which the generic parser above cannot express.
if [[ "${ARGS["hint_mode"]}" == "retrieval" && "${ARGS["info_docs"]}" == "none" ]]; then
    echo "Error: --hint_mode retrieval requires --info_docs (run scripts/vlm/build_info.sh first)."; exit 1
fi
if [[ "${ARGS["hint_mode"]}" == "init_state" && "${ARGS["insights_paths"]}" == "none" ]]; then
    echo "Error: --hint_mode init_state requires --insights_paths (run scripts/vlm/build_info.sh --stage a)."; exit 1
fi

regenerate_flag=""
if [[ "${ARGS["regenerate"]}" == "true" ]]; then regenerate_flag="--regenerate"; fi

docs_arg=""
if [[ "${ARGS["info_docs"]}" != "none" ]]; then docs_arg="--info_docs ${ARGS["info_docs"]}"; fi

insights_arg=""
if [[ "${ARGS["insights_paths"]}" != "none" ]]; then insights_arg="--insights_paths ${ARGS["insights_paths"]}"; fi

hint_model_arg=""
if [[ "${ARGS["hint_vlm_model"]}" != "none" ]]; then hint_model_arg="--hint_vlm_model ${ARGS["hint_vlm_model"]}"; fi

hint_kind_arg=""
if [[ "${ARGS["hint_vlm_kind"]}" != "none" ]]; then hint_kind_arg="--hint_vlm_kind ${ARGS["hint_vlm_kind"]}"; fi

# run_benchmark_info.py keeps the python-level flag as --mode; only the shell vocabulary is
# disambiguated, so the CSV/log naming below still matches the runner's own.
common="python run_benchmark_info.py --game ${ARGS["game"]} --executor ${ARGS["executor"]} --mode ${ARGS["hint_mode"]} --save_video True --max_resets ${ARGS["max_resets"]} --max_steps ${ARGS["max_steps"]} --max_concurrency ${ARGS["max_concurrency"]} --executor_vlm_model ${ARGS["executor_vlm_model"]} --executor_vlm_kind ${ARGS["executor_vlm_kind"]} $docs_arg $insights_arg $hint_model_arg $hint_kind_arg $regenerate_flag"

model_save_name="${ARGS["executor_vlm_model"]##*/}"
mkdir -p "results/benchmark/${ARGS["game"]}/"

$common
