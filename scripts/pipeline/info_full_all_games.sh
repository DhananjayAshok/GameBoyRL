#!/usr/bin/env bash
# Runs info_full.sh for every game that has usable inputs — the whole context-engineering
# arm swept across the dataset. The "all" wrapper around info_full.sh, in the same spirit as
# curiosity_and_zeroshot_all.sh wrapping curiosity_and_zeroshot.sh.
#
# A game is INCLUDED when it has, for --model_name:
#   - at least one source vertical on disk (zeroshot attempts and/or curiosity annotations), and
#   - benchmark tasks defined under GameBoyWorlds/benchmark/tests/.
# Each game gets the --mode its own data supports: both when both verticals exist, otherwise
# the one that does. Demanding both would skip most games, since most have only one.
#
# One game's failure does NOT abort the sweep — a broken game is reported and the run moves
# on, because the expensive thing here is the games that would have worked.
#
# Assumes a VLM server is already serving --model_name (does NOT start one), matching full.sh.
# Use --dry_run true to print the plan (games, modes, resolved stems) without spending
# anything; do that first, the full sweep is many GPU-hours.

source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
REQUIRED_ARGS=()

# Inherited wholesale from info_full's arrays in utils.sh, plus this script's own
# game-selection flags (--games, --skip_games, --dry_run) and an accepted-and-ignored --game.
# See INFO_FULL_ALL_GAMES_DEFAULTS: --mode is deliberately not settable here, since each
# game's mode is discovered from what it has on disk.
#
# Nothing is redeclared: the sweep and info_full.sh must not be able to disagree about a
# default, or a flag the sweep hardcodes silently overrides the per-game script's value.
populate_array INFO_FULL_ALL_GAMES_ESSENTIALS REQUIRED_ARGS   # model_name, vlm_kind, run_name
populate_dict INFO_FULL_ALL_GAMES_DEFAULTS ARGS

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

run_name="${ARGS["run_name"]}"
executor="${ARGS["executor"]}"
model_save_name="${ARGS["model_name"]##*/}"

# --- Which games to sweep -------------------------------------------------------------
if [[ "${ARGS["games"]}" == "auto" ]]; then
    mapfile -t candidate_games < <(
        for d in "$storage_dir"/proposed_tasks/*/; do
            game=$(basename "$d")
            [[ -d "$d/$model_save_name" ]] && echo "$game"
        done | sort -u
    )
else
    IFS=',' read -ra candidate_games <<< "${ARGS["games"]}"
fi

IFS=',' read -ra skip_list <<< "${ARGS["skip_games"]}"

games=()
modes=()
skipped=()
for game in "${candidate_games[@]}"; do
    [[ -z "$game" ]] && continue

    explicitly_skipped=false
    for s in "${skip_list[@]}"; do
        if [[ "$game" == "$s" ]]; then explicitly_skipped=true; break; fi
    done
    if [[ "$explicitly_skipped" == true ]]; then
        skipped+=("$game (--skip_games)"); continue
    fi

    # A game with no benchmark tasks can be built but never scored, so it is not worth the
    # build spend.
    if ! grep -qh "^$game," "$PROJECT_ROOT"/GameBoyWorlds/benchmark/tests/*.csv 2>/dev/null; then
        skipped+=("$game (no benchmark tasks)"); continue
    fi

    sources=$(info_available_sources "$game" "$model_save_name" "$run_name" "$executor")
    if [[ -z "$sources" ]]; then
        skipped+=("$game (no zeroshot or curiosity inputs)"); continue
    fi

    games+=("$game")
    modes+=("$(info_sources_to_mode "$sources")")
done

echo ""
echo "########## Sweep plan ##########"
printf "%-42s %-16s %s\n" "GAME" "MODE" "SOURCES"
for i in "${!games[@]}"; do
    sources=$(info_available_sources "${games[$i]}" "$model_save_name" "$run_name" "$executor")
    printf "%-42s %-16s %s\n" "${games[$i]}" "${modes[$i]}" "$sources"
done
if [[ ${#skipped[@]} -gt 0 ]]; then
    echo ""
    echo "Skipped:"
    for s in "${skipped[@]}"; do echo "  - $s"; done
fi
echo ""
echo "${#games[@]} game(s) to run."

if [[ ${#games[@]} -eq 0 ]]; then
    echo "Nothing to do."; exit 0
fi

if [[ "${ARGS["dry_run"]}" == "true" ]]; then
    echo ""
    echo "Dry run — resolved inputs per game, nothing executed:"
    for i in "${!games[@]}"; do
        echo ""
        echo "  ${games[$i]}  (--mode ${modes[$i]})"
        for source in $(info_available_sources "${games[$i]}" "$model_save_name" "$run_name" "$executor"); do
            stem=$(info_source_stem "${games[$i]}" "$model_save_name" "$run_name" "$executor" "$source")
            echo "    $source stem     : $stem"
            echo "    $source info_dir : $(info_dir_for_stem "$stem" "$model_save_name" "$executor")"
        done
    done
    exit 0
fi

# --- Sweep ----------------------------------------------------------------------------
failed=()
for i in "${!games[@]}"; do
    game="${games[$i]}"
    mode="${modes[$i]}"

    echo ""
    echo "##########################################################################"
    echo "# [$((i + 1))/${#games[@]}] $game  (--mode $mode)"
    echo "##########################################################################"

    ARGS["game"]="$game"
    ARGS["mode"]="$mode"
    flags=$(args_to_flags_subset ARGS INFO_FULL_ARG_KEYS)

    # Deliberately not `|| exit 1`: one game failing (a corrupt pkl, a game whose documents
    # came out empty) must not throw away the remaining games' runs.
    if ! bash scripts/pipeline/info_full.sh $flags; then
        echo "WARNING: $game failed; continuing with the remaining games."
        failed+=("$game")
    fi
done

echo ""
echo "########## Sweep complete ##########"
echo "Ran ${#games[@]} game(s); ${#failed[@]} failed."
if [[ ${#failed[@]} -gt 0 ]]; then
    for f in "${failed[@]}"; do echo "  FAILED: $f"; done
fi
echo "Diagnostics: $results_dir/debug/<game>/info_*/info/report.md"
echo "Results:     $results_dir/benchmark/<game>/"
