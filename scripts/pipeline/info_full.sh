#!/usr/bin/env bash
# End-to-end driver for the context-engineering arm: distils a game's (task, trajectory)
# pairs into info documents, then benchmarks a frozen VLM with hints drawn from them.
#
# The parallel to full.sh. Where full.sh bakes knowledge into weights (collect -> fine-tune),
# this bakes it into an inspectable markdown document (build -> hint at inference). Both
# consume the same upstream data and write the same benchmark CSV shape, so their results
# are directly comparable — and combinable, since a fine-tuned checkpoint can still be given
# hints (point --model_name at the served checkpoint).
#
# Assumes a VLM server is already serving --model_name (does NOT start one) — bring up the
# server (e.g. scripts/core/serve_vllm.sh, or any `vllm serve` equivalent) before calling
# this, exactly as full.sh requires.
#
# Nothing here trains. The frozen-model arm's whole claim is zero weight updates, so the
# only cost is VLM calls: ~1 per (task, trajectory) pair to build, then a bounded number
# per benchmark episode to select and write each hint.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

# The key set and every default live in utils.sh (INFO_FULL_ESSENTIALS / INFO_FULL_DEFAULTS),
# built as the union of the two stages this script calls. Nothing is redeclared here: a value
# spelled out in both places is a value that drifts, and the sweep wrapper inherits the same
# arrays, so a default changed in utils.sh changes for both entry points at once.
populate_array INFO_FULL_ESSENTIALS REQUIRED_ARGS   # game, model_name, vlm_kind, run_name
populate_dict INFO_FULL_DEFAULTS ARGS

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

# Stage 1: build an info document per source. Every stage here early-returns on existing
# output, so re-running after a timeout resumes rather than re-paying.
if [[ "${ARGS["do_build"]}" == "true" ]]; then
    echo ""
    echo "########## Stage 1: build info documents (mode=$mode, stage=${ARGS["stage"]}) ##########"
    # Each stage runs at its own width (see the split in INFO_FULL_DEFAULTS): building
    # documents and sampling benchmark episodes do not want the same concurrency, and one
    # shared --max_concurrency would force them to.
    ARGS["max_concurrency"]="${ARGS["build_max_concurrency"]}"
    build_flags=$(args_to_flags_subset ARGS BUILD_INFO_ALL_ARG_KEYS)
    bash scripts/pipeline/build_info_all.sh $build_flags || exit 1
else
    echo "Skipping build (--do_build false); expecting documents to exist already."
fi

# Stage 2: benchmark the plan arm, plus the no-knowledge baseline the numbers are read against.
if [[ "${ARGS["do_benchmark"]}" == "true" ]]; then
    echo ""
    echo "########## Stage 2: benchmark plan arm (knowledge_mode=${ARGS["knowledge_mode"]}) ##########"
    ARGS["max_concurrency"]="${ARGS["bench_max_concurrency"]}"
    bench_flags=$(args_to_flags_subset ARGS BENCHMARK_INFO_ALL_ARG_KEYS)
    bash scripts/pipeline/benchmark_info_all.sh $bench_flags || exit 1
else
    echo "Skipping benchmark (--do_benchmark false)."
fi

echo ""
echo "########## Done ##########"
echo "Diagnostics: $results_dir/debug/$game/info_*/info/report.md"
echo "Results:     $results_dir/benchmark/$game/"
echo ""
echo "Read the diagnostics first: a document whose stage-A yield was near-zero, or whose"
echo "benchmark init_state coverage is thin, scores like the baseline for reasons that have"
echo "nothing to do with plan quality."
