source scripts/core/utils.sh
source configs/config.env

games=("deja_vu_2" "legend_of_zelda_the_oracle_of_seasons" "harvest_moon_3")
#games=("harvest_moon_1" "harvest_moon_2" "harvest_moon_3")
#games=("survival_kids_1")
#games=("pokemon_red" "legend_of_zelda_links_awakening" "sword_of_hope_1" "harvest_moon_1" "bomberman_pocket" "deja_vu_1")

if [ -z "$model" ]; then
    echo "ERROR: 'model' variable is not set. Export model=<org/name> before running. Aborting."
    exit 1
fi

# The nine executor arms are <action>_<history>. Listed rather than generated so a sweep can
# be narrowed without editing the registry.
executors=("single_none" "single_actions" "single_visual" \
           "scored_none" "scored_actions" "scored_visual" \
           "sequence_none" "sequence_actions" "sequence_visual")


supervisors=("dummy" "revision" "subgoal")


for game in "${games[@]}"; do
    echo ""
    echo "=== game: $game ==="
    for executor in "${executors[@]}"; do
        for supervisor in "${supervisors[@]}"; do
            echo "--- $game | $executor | $supervisor"
            bash scripts/benchmark/run_benchmark.sh \
                --game "$game" \
                --supervisor "$supervisor" \
                --executor "$executor" \
                --executor_vlm_model "$model" \
                --executor_vlm_kind vllm \
                || echo "FAILED: $game | $executor | $supervisor"
        done
    done
done

echo "DONE ALL"
