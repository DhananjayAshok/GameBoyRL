#!/usr/bin/env bash
# Benchmarks the context-engineering arm: runs the hinted benchmark for each --hint_mode
# over the documents build_info_all.sh produced, plus the no-hint baseline the results are
# only interpretable against.
#
# Paths are re-derived from the same utils.sh helpers build_info_all.sh used, never passed
# between the two scripts (the merged_dataset_dir pattern).
#
# --mode      which source verticals to draw knowledge from (curiosity_only/zeroshot_only/both).
#             Several sources are unioned into one comma-separated list, and each entry keeps
#             a provenance label so the hint writer knows verified solutions from exploration.
# --hint_mode which test-time selection path to run:
#               retrieval   ask per entry whether it fits this task and screen (needs info.md)
#               init_state  key off the episode's init state (needs only insights.jsonl)
#               both        run each in turn — the pair is what decomposes "do hints help?"
#                           into "are the insights good?" and "is selection working?"
#
# Assumes a VLM server is already serving --model_name (does NOT start one), matching full.sh.
#
# --executor  ONE knob, used for both halves: the attempts/curiosity dirs the documents were
#             built from, and the executor run at test time. The baseline below must be the
#             same executor as the hinted runs or the delta is not attributable to hints.
#
# Results: results/benchmark/<game>/info_<hint_mode>_<executor>_<model>.csv
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
hint_mode="${ARGS["hint_mode"]}"
model_name="${ARGS["model_name"]}"
model_save_name="${model_name##*/}"

sources=$(info_mode_sources "$mode")
if [[ -z "$sources" ]]; then
    echo "Error: unknown --mode '$mode'. Choose one of: curiosity_only, zeroshot_only, both."
    exit 1
fi

case "$hint_mode" in
    retrieval|init_state) hint_modes="$hint_mode" ;;
    both)                 hint_modes="init_state retrieval" ;;
    *) echo "Error: unknown --hint_mode '$hint_mode'. Choose one of: retrieval, init_state, both."
       exit 1 ;;
esac
echo "Mode '$mode' -> sources: $sources"
echo "Hint mode '$hint_mode' -> runs: $hint_modes"

# Collect the artifact paths for every selected source, comma-joined for --insights_paths /
# --info_docs. Missing artifacts are named per source rather than reported as one opaque
# failure, since with --mode both it is usually only one of the two that was never built.
insights_paths=""
info_docs=""
for source in $sources; do
    stem=$(info_source_stem "$game" "$model_save_name" "$run_name" "$executor" "$source")
    info_dir=$(info_dir_for_stem "$stem" "$model_save_name" "$executor")

    insights="$info_dir/insights.jsonl"
    doc="$info_dir/info.md"

    if [[ ! -f "$insights" ]]; then
        echo "Error: $source insights not found at $insights"
        echo "  Produced by: scripts/pipeline/build_info_all.sh --mode $mode"
        exit 1
    fi
    insights_paths+="${insights_paths:+,}$insights"

    if [[ -f "$doc" ]]; then
        info_docs+="${info_docs:+,}$doc"
    elif [[ "$hint_modes" == *retrieval* ]]; then
        echo "Error: $source info.md not found at $doc, but --hint_mode includes retrieval."
        echo "  Produced by: scripts/pipeline/build_info_all.sh --mode $mode --stage all"
        echo "  (--stage a builds only insights.jsonl, which is enough for init_state.)"
        exit 1
    fi
    echo "  $source -> $info_dir"
done

hint_model_arg=""
if [[ "${ARGS["hint_vlm_model"]}" != "none" ]]; then
    hint_model_arg="--hint_vlm_model ${ARGS["hint_vlm_model"]}"
fi
hint_kind_arg=""
if [[ "${ARGS["hint_vlm_kind"]}" != "none" ]]; then
    hint_kind_arg="--hint_vlm_kind ${ARGS["hint_vlm_kind"]}"
fi

for run_hint_mode in $hint_modes; do
    echo ""
    echo "=== Benchmark: --hint_mode $run_hint_mode ==="
    bash scripts/benchmark_info.sh \
        --game "$game" \
        --hint_mode "$run_hint_mode" \
        --insights_paths "$insights_paths" \
        --info_docs "${info_docs:-none}" \
        --executor "$executor" \
        --executor_vlm_model "$model_name" \
        --executor_vlm_kind "${ARGS["vlm_kind"]}" \
        --max_steps "${ARGS["max_steps"]}" \
        --max_resets "${ARGS["max_resets"]}" \
        --max_concurrency "${ARGS["max_concurrency"]}" \
        --regenerate "${ARGS["regenerate"]}" \
        $hint_model_arg $hint_kind_arg || exit 1
    echo "  -> $results_dir/benchmark/$game/info_${run_hint_mode}_${executor}_${model_save_name}.csv"
done

# The no-hint run on the SAME executor and model. Without it the hinted numbers above are
# just success counts with nothing to be better than, which is the whole question this arm
# asks. Same CSV shape, different filename, so viz.py reads them side by side.
if [[ "${ARGS["baseline"]}" == "true" ]]; then
    echo ""
    echo "=== Benchmark: no-hint baseline ==="
    bash scripts/benchmark.sh \
        --game "$game" \
        --executor "$executor" \
        --executor_vlm_model "$model_name" \
        --executor_vlm_kind "${ARGS["vlm_kind"]}" \
        --max_steps "${ARGS["max_steps"]}" \
        --max_resets "${ARGS["max_resets"]}" \
        --regenerate "${ARGS["regenerate"]}" || exit 1
    echo "  -> $results_dir/benchmark/$game/${executor}_${model_save_name}.csv"
fi

echo ""
echo "Done. Compare the info_* CSVs against the baseline in $results_dir/benchmark/$game/"
