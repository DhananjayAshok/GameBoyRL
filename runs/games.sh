#!/usr/bin/env bash
# Game lists shared by every driver under runs/. Sourced, never executed.
#
# These are the record of what a sweep covers. Narrow a run with --games on the driver
# rather than by commenting entries out here.

# The ten-title benchmark sweep.
BENCH_GAMES=(
    "bomberman_pocket"
    "bomberman_quest"
    "deja_vu_1"
    "deja_vu_2"
    "legend_of_zelda_links_awakening"
    "legend_of_zelda_the_oracle_of_seasons"
    "pokemon_crystal"
    "pokemon_red"
    "sword_of_hope_1"
    "sword_of_hope_2"
)

# Titles with zeroshot attempts to practice from.
PRACTICE_GAMES=(
    "deja_vu_1"
    "pokemon_red"
    "sword_of_hope_1"
    "legend_of_zelda_links_awakening"
    "bomberman_quest"
)

# Titles with practice datasets to fine-tune on. Same set as PRACTICE_GAMES today; kept
# separate because the two stages gate on different artifacts.
TRAIN_GAMES=("${PRACTICE_GAMES[@]}")

# Pokemon titles the strategist plays.
POKEMON_GAMES=(
    "pokemon_red"
    "pokemon_crystal"
    "pokemon_brown"
    "pokemon_prism"
)

# resolve_games <default_array_name> <comma_separated_or_none>
#
# Echoes one game per line: the named default list when the second argument is `none`,
# otherwise the comma-separated selection, validated against the default list so a typo
# fails here instead of after the first arm has already run.
function resolve_games() {
    local -n default_list="$1"
    local requested="$2"
    if [[ "$requested" == "none" ]]; then
        printf '%s\n' "${default_list[@]}"
        return 0
    fi
    local -a chosen
    IFS=',' read -ra chosen <<< "$requested"
    local game found
    for game in "${chosen[@]}"; do
        found=false
        for known in "${default_list[@]}"; do
            if [[ "$game" == "$known" ]]; then found=true; break; fi
        done
        if [[ "$found" == false ]]; then
            echo "resolve_games: '$game' is not in $1: ${default_list[*]}" >&2
            return 1
        fi
        echo "$game"
    done
}
