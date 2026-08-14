#!/usr/bin/env bash
# Benchmarks the context-engineering arm: runs the plan benchmark for each --knowledge_mode
# over the documents build_info_all.sh produced, plus the no-knowledge baseline the results
# are only interpretable against.
#
# Paths are re-derived from the same utils.sh helpers build_info_all.sh used, never passed
# between the two scripts (the info_source_stem pattern).
#
# --mode      which source verticals to draw knowledge from (curiosity_only/zeroshot_only/both).
#             Several sources are unioned into one comma-separated list, and each entry keeps
#             a provenance label so the planner knows verified solutions from exploration.
#             Only --knowledge_mode retrieval reads them.
# --knowledge_mode  where the planner's knowledge comes from:
#               retrieval   the documents build_info_all produced (needs info.json)
#               parametric  a document the model writes from its own priors, from the game's
#                           name alone — needs no built artifacts
#               both        run each in turn. This is the comparison worth having: everything
#                           after the document is identical, so a retrieval run that does not
#                           beat parametric has not shown that distillation bought anything.
#
# Assumes a VLM server is already serving --model_name (does NOT start one), matching full.sh.
#
# --executor  ONE knob, used for both halves: the attempts/curiosity dirs the documents were
#             built from, and the executor run at test time. The baseline below must be the
#             same executor as the plan runs or the delta is not attributable to the plan.
#
# Results: results/benchmark/<game>/info_subgoal_<knowledge_mode>_<executor>_<model>.csv
#          results/benchmark/<game>/<executor>_<model>.csv               (baseline)

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array BENCHMARK_INFO_ALL_ESSENTIALS REQUIRED_ARGS
populate_dict BENCHMARK_INFO_ALL_DEFAULTS ARGS

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
run_name="${ARGS["run_name"]}"
executor="${ARGS["executor"]}"
mode="${ARGS["mode"]}"
knowledge_mode="${ARGS["knowledge_mode"]}"
model_name="${ARGS["model_name"]}"
model_save_name=$(model_save_name "$model_name")

sources=$(info_mode_sources "$mode")
if [[ -z "$sources" ]]; then
    echo "Error: unknown --mode '$mode'. Choose one of: curiosity_only, zeroshot_only, both."
    exit 1
fi

case "$knowledge_mode" in
    retrieval|parametric) knowledge_modes="$knowledge_mode" ;;
    both)                 knowledge_modes="parametric retrieval" ;;
    *) echo "Error: unknown --knowledge_mode '$knowledge_mode'. Choose one of: retrieval, parametric, both."
       exit 1 ;;
esac
echo "Mode '$mode' -> sources: $sources"
echo "Knowledge mode '$knowledge_mode' -> runs: $knowledge_modes"

# Collect the document path for every selected source, comma-joined for --info_docs. Missing
# artifacts are named per source rather than reported as one opaque failure, since with
# --mode both it is usually only one of the two that was never built.
#
# Skipped entirely under parametric-only: that mode reads no built artifacts, so requiring
# them would make the control impossible to run on a game nothing has been built for — which
# is exactly the case it is most useful in.
info_docs=""
if [[ "$knowledge_modes" == *retrieval* ]]; then
    for source in $sources; do
        info_dir=$(path_of source_info_dir --game "$game" --model_name "$model_name" \
                           --run_name "$run_name" --executor "$executor" --source "$source")
        doc="$info_dir/info.json"

        if [[ ! -f "$doc" ]]; then
            echo "Error: $source info.json not found at $doc, but --knowledge_mode includes retrieval."
            echo "  Produced by: scripts/pipeline/build_info_all.sh --mode $mode --stage all"
            echo "  (--stage a builds only insights.jsonl, which no benchmark mode reads any more.)"
            exit 1
        fi
        info_docs+="${info_docs:+,}$doc"
        echo "  $source -> $info_dir"
    done
fi

supervisor_model_arg=""
if [[ "${ARGS["supervisor_vlm_model"]}" != "none" ]]; then
    supervisor_model_arg="--supervisor_vlm_model ${ARGS["supervisor_vlm_model"]}"
fi
supervisor_kind_arg=""
if [[ "${ARGS["supervisor_vlm_kind"]}" != "none" ]]; then
    supervisor_kind_arg="--supervisor_vlm_kind ${ARGS["supervisor_vlm_kind"]}"
fi

for run_knowledge_mode in $knowledge_modes; do
    echo ""
    echo "=== Benchmark: plan arm, --knowledge_mode $run_knowledge_mode ==="
    bash scripts/benchmark_plan.sh \
        --game "$game" \
        --knowledge_mode "$run_knowledge_mode" \
        --info_docs "${info_docs:-none}" \
        --executor "$executor" \
        --executor_vlm_model "$model_name" \
        --executor_vlm_kind "${ARGS["vlm_kind"]}" \
        --max_steps "${ARGS["max_steps"]}" \
        --max_concurrency "${ARGS["max_concurrency"]}" \
        --regenerate "${ARGS["regenerate"]}" \
        $supervisor_model_arg $supervisor_kind_arg || exit 1
    echo "  -> $results_dir/benchmark/$game/info_subgoal_${run_knowledge_mode}_${executor}_${model_save_name}.csv"
done

# The no-knowledge run on the SAME executor and model. Without it the plan numbers above are
# just success counts with nothing to be better than, which is the whole question this arm
if [[ "${ARGS["baseline"]}" == "true" ]]; then
    echo ""
    echo "=== Benchmark: no-knowledge baseline ==="
    bash scripts/benchmark.sh \
        --game "$game" \
        --executor "$executor" \
        --executor_vlm_model "$model_name" \
        --executor_vlm_kind "${ARGS["vlm_kind"]}" \
        --max_steps "${ARGS["max_steps"]}" \
        --regenerate "${ARGS["regenerate"]}" || exit 1
    echo "  -> $results_dir/benchmark/$game/${executor}_${model_save_name}.csv"
fi

echo ""
echo "Done. Compare the info_subgoal_* CSVs against the baseline in $results_dir/benchmark/$game/"
