#!/usr/bin/env bash
# Builds an info document for every source vertical a --mode selects, then writes the
# diagnostic report for each. This is the build half of the context-engineering arm — the
# parallel to curiosity_and_zeroshot_all.sh in the fine-tuning arm.
#
# Both verticals terminate in the same shape ({group: task} + {group: trajectory} under one
# stem), so the same build script serves both; only --trajectory_path differs. build_info
# always writes beside its own input, so each source gets its own info dir and neither can
# clobber the other.
#
# Assumes a VLM server is already serving --model_name (does NOT start one), matching full.sh.
#
# Output per source, under <dirname(stem)>/info_docs/:
#   insights.jsonl   stage A leaves — the merge tree's input, and what debug.py info reports
#                    on. No benchmark mode reads it directly any more.
#   frames/          representative frames, for visual matching
#   merge/, info.json  stage B — required by benchmark_info_all.sh --knowledge_mode retrieval,
#                    so --stage all is mandatory if you intend to benchmark

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array BUILD_INFO_ALL_ESSENTIALS REQUIRED_ARGS
populate_dict BUILD_INFO_ALL_DEFAULTS ARGS

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
controller_variant="${ARGS["controller_variant"]}"
mode="${ARGS["mode"]}"
model_save_name=$(model_save_name "${ARGS["model_name"]}")

sources=$(info_mode_sources "$mode")
if [[ -z "$sources" ]]; then
    echo "Error: unknown --mode '$mode'. Choose one of: curiosity_only, zeroshot_only, both."
    exit 1
fi
echo "Mode '$mode' -> sources: $sources"

# Validate every source before spending a single VLM call on any of them: a --mode both run
# that dies on the second source after paying for the first is the expensive failure here.
for source in $sources; do
    stem=$(path_of info_source_stem --game "$game" --model_name "${ARGS["model_name"]}" \
                   --run_name "$run_name" --executor "$executor" \
                   --controller_variant "$controller_variant" --source "$source")
    for suffix in json pkl; do
        if [[ ! -f "$stem.$suffix" ]]; then
            echo "Error: $source input missing at $stem.$suffix"
            case "$source" in
                zeroshot)  echo "  Produced by: scripts/vlm/attempt_tasks.sh (via propose_and_attempt_all.sh)" ;;
                curiosity) echo "  Produced by: scripts/vlm/infer_tasks.sh (via curiosity_all_tasks.sh)" ;;
            esac
            exit 1
        fi
    done
    echo "  $source -> $stem"
done

for source in $sources; do
    stem=$(path_of info_source_stem --game "$game" --model_name "${ARGS["model_name"]}" \
                   --run_name "$run_name" --executor "$executor" \
                   --controller_variant "$controller_variant" --source "$source")
    info_dir=$(path_of source_info_dir --game "$game" --model_name "${ARGS["model_name"]}" \
                       --run_name "$run_name" --executor "${ARGS["executor"]}" \
                       --controller_variant "$controller_variant" --source "$source")

    echo ""
    echo "=== Building info document: $source ==="
    ARGS["trajectory_path"]="$stem"
    # Recorded as the document's provenance rather than re-derived from the output path by
    # every reader. It is known here: the vertical is the loop variable.
    ARGS["source"]="$source"
    flags=$(args_to_flags_subset ARGS BUILD_INFO_ARG_KEYS)
    bash scripts/vlm/build_info.sh $flags || exit 1
    echo "  -> $info_dir"
done

# The diagnostic report per source: yield funnel, benchmark init_state coverage, and (when
# stage B ran) the merge curve, insight drift and match audit. Offline and free to re-run —
# read this BEFORE benchmarking, since a document that came out empty scores like the
# baseline for reasons that have nothing to do with hint quality.
if [[ "${ARGS["do_debug"]}" == "true" ]]; then
    for source in $sources; do
        # debug.py's --source names the paths.py accessor, and its vocabulary for the
        # zeroshot vertical is "attempt" (the artifacts are attempt_tasks output).
        debug_source="$source"
        if [[ "$source" == "zeroshot" ]]; then debug_source="attempt"; fi

        echo ""
        echo "=== Diagnostics: $source ==="
        # Separate --output_dir per source, or the second report overwrites the first.
        # NOT fatal: the documents are the product, this is a report about them. A game
        # whose documents built fine must still reach its benchmark.
        if ! python debug.py \
            --game "$game" \
            --run_name "$run_name" \
            --executor "$executor" \
            --output_dir "$results_dir/debug/$game/info_$source" \
            info \
            --model_name "${ARGS["model_name"]}" \
            --source "$debug_source"; then
            echo "WARNING: diagnostics failed for $source; documents are built, continuing."
        else
            echo "  -> $results_dir/debug/$game/info_$source/info/report.md"
        fi
    done
fi

echo ""
echo "Done. Info documents built for: $sources"
