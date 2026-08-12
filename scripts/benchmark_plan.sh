#!/usr/bin/env bash
# Runs the benchmark suite with an info-document plan (the context-engineering arm).
# A sibling of benchmark.sh: same game/executor/model args, but each task goes through an
# InfoPlanSupervisor that retrieves knowledge, writes a plan from it, and supervises the
# executor through that plan one step at a time.
#
#   --knowledge_mode retrieval   plans from documents distilled out of real trajectories.
#                                Needs --info_docs (from scripts/vlm/build_info.sh, --stage all).
#   --knowledge_mode parametric  plans from a document the model writes about the game from
#                                its own priors, given only the game's name. Needs no
#                                artifacts at all; the document is generated on first use and
#                                cached under <storage_dir>/parametric_docs/<game>/<model>/.
#
# The pair is the control for the whole vertical: everything after the document — relevance,
# filtering, planning, the step loop — is identical, so a retrieval run that does not beat a
# parametric one has not shown that distilling trajectories bought anything.
#
# The flag is --knowledge_mode, not --mode: across the pipeline scripts --mode always means
# the source selection (curiosity_only / zeroshot_only / both), and the two are orthogonal.
#
# --info_docs takes comma-separated paths, so several sources can be unioned in one run.
# Results land in results/benchmark/<game>/info_plan_<knowledge_mode>_<executor>_<model>.csv.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

# Script-specific defaults and required args
declare -A ARGS
ARGS["executor"]="simple"
ARGS["executor_vlm_model"]="Qwen/Qwen3-VL-8B-Instruct"   # use "none" for absent optionals, never ""
ARGS["executor_vlm_kind"]="huggingface"   # use "none" for absent optionals, never ""
ARGS["knowledge_mode"]="retrieval"
ARGS["info_docs"]=none
ARGS["supervisor_vlm_model"]=none
ARGS["supervisor_vlm_kind"]=none
ARGS["max_concurrency"]="8"
ARGS["max_steps"]="50"
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
# --knowledge_mode, which the generic parser above cannot express. parametric needs nothing
# — that is the point of it — so it has no check.
if [[ "${ARGS["knowledge_mode"]}" == "retrieval" && "${ARGS["info_docs"]}" == "none" ]]; then
    echo "Error: --knowledge_mode retrieval requires --info_docs (run scripts/vlm/build_info.sh first)."; exit 1
fi

regenerate_flag=""
if [[ "${ARGS["regenerate"]}" == "true" ]]; then regenerate_flag="--regenerate"; fi

docs_arg=""
if [[ "${ARGS["info_docs"]}" != "none" ]]; then docs_arg="--info_docs ${ARGS["info_docs"]}"; fi

# The one model the supervisor reasons with, as distinct from the executor's. Defaults to the
# executor's when unset. Under --knowledge_mode parametric this is also the model whose priors
# become the document, and the cache path is keyed on it — there is no third model anywhere.
#
# A GROUP option, not an arm option: every arm's supervisor uses it, so it goes before the
# subcommand word below.
supervisor_model_arg=""
if [[ "${ARGS["supervisor_vlm_model"]}" != "none" ]]; then supervisor_model_arg="--supervisor_vlm_model ${ARGS["supervisor_vlm_model"]}"; fi

supervisor_kind_arg=""
if [[ "${ARGS["supervisor_vlm_kind"]}" != "none" ]]; then supervisor_kind_arg="--supervisor_vlm_kind ${ARGS["supervisor_vlm_kind"]}"; fi

# run_benchmark.py is a click group: options shared by every arm must precede the subcommand
# word, and the arm's own options must follow it. Splitting the two here rather than building
# one flag string is what keeps that ordering correct.
#
# The python-level flag is --mode; only the shell vocabulary is disambiguated to
# knowledge_mode, so the CSV/log naming below still matches the runner's own.
group_flags="--game ${ARGS["game"]} --executor ${ARGS["executor"]} --save_video True --max_steps ${ARGS["max_steps"]} --executor_vlm_model ${ARGS["executor_vlm_model"]} --executor_vlm_kind ${ARGS["executor_vlm_kind"]} $supervisor_model_arg $supervisor_kind_arg $regenerate_flag"
arm_flags="--mode ${ARGS["knowledge_mode"]} --max_concurrency ${ARGS["max_concurrency"]} $docs_arg"
common="python run_benchmark.py $group_flags plan $arm_flags"

$common
