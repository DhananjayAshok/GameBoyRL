#!/usr/bin/env bash
# scripts/run_benchmark.sh's info_subgoal_retrieval arm, with --info_docs discovered rather
# than typed: --game is resolved to its train games (the titles that declare train states, so
# the only ones with source data), and every info document on disk for those is comma-joined.
#
# --docs_model / --docs_executor / --docs_run_name select WHICH documents to read; they are
# inputs on disk and are independent of the model being benchmarked (--executor_vlm_model).

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

populate_array RUN_BENCHMARK_INFO_RETRIEVAL_ESSENTIALS REQUIRED_ARGS
populate_dict RUN_BENCHMARK_INFO_RETRIEVAL_DEFAULTS ARGS

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
docs_model="${ARGS["docs_model"]}"
docs_executor="${ARGS["docs_executor"]}"
docs_controller_variant="${ARGS["docs_controller_variant"]}"
docs_run_name="${ARGS["docs_run_name"]}"

wanted_sources=$(info_mode_sources "${ARGS["docs_mode"]}")
if [[ -z "$wanted_sources" ]]; then
    echo "Error: unknown --docs_mode '${ARGS["docs_mode"]}'. Choose one of: curiosity_only, zeroshot_only, both."
    exit 1
fi
echo "docs_mode '${ARGS["docs_mode"]}' -> sources: $wanted_sources"

# path_of's own exit only kills the subshell here, so an empty list is checked for below
# rather than assumed to be impossible.
mapfile -t docs_games < <(path_of train_games --game "$game")
if [[ ${#docs_games[@]} -eq 0 ]]; then
    echo "Error: no train games for $game (path_of train_games returned nothing)."
    exit 1
fi
echo "Train games for $game: ${docs_games[*]}"

# Requested sources intersected with what has build_info inputs on disk: a game may declare
# --docs_mode both and have only one vertical, which is the common case, not an error.
info_docs=""
for docs_game in "${docs_games[@]}"; do
    available=$(path_of_allow_empty info_available_sources --game "$docs_game" \
                    --model_name "$docs_model" --run_name "$docs_run_name" \
                    --executor "$docs_executor" \
                    --controller_variant "$docs_controller_variant")
    for source in $wanted_sources; do
        if [[ ! " $available " =~ " $source " ]]; then
            echo "  $docs_game/$source -> no build_info input on disk, skipping"
            continue
        fi
        doc=$(path_of info_doc --game "$docs_game" --model_name "$docs_model" \
                  --run_name "$docs_run_name" --executor "$docs_executor" \
                  --controller_variant "$docs_controller_variant" --source "$source")
        if [[ -f "$doc" ]]; then
            info_docs+="${info_docs:+,}$doc"
            echo "  $docs_game/$source -> $doc"
        else
            echo "  $docs_game/$source -> MISSING $doc"
        fi
    done
done

# Hard error, not a skip: --info_docs is required by the info_subgoal_retrieval subcommand,
# and running the arm with nothing to plan from is not the experiment.
if [[ -z "$info_docs" ]]; then
    echo "Error: no info documents on disk for ${docs_games[*]}."
    echo "  Looked under --docs_model $docs_model --docs_run_name $docs_run_name --docs_executor $docs_executor"
    echo "  A MISSING line above with insights.jsonl beside it means stage B has not run under"
    echo "  the current JSON format: scripts/pipeline/build_info_all.sh --stage b"
    exit 1
fi

ARGS["supervisor"]="info_subgoal_retrieval"
ARGS["info_docs"]="$info_docs"

# Subset against run_benchmark.sh's own key list, which drops the three docs_* lookup flags it
# does not accept.
arg_string=$(args_to_flags_subset ARGS RUN_BENCHMARK_ARG_KEYS)
bash scripts/run_benchmark.sh $arg_string
