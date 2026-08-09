source scripts/core/utils.sh
games=("pokemon_red" "sword_of_hope_2" "deja_vu_2" "legend_of_zelda_the_oracle_of_seasons" "harvest_moon_3" "bomberman_max" "survival_kids_2" "runes_of_virtue_2")
#games=("harvest_moon_1" "harvest_moon_2" "harvest_moon_3")
#games=("survival_kids_1")
#games=("pokemon_red" "legend_of_zelda_links_awakening" "sword_of_hope_1" "harvest_moon_1" "bomberman_pocket" "deja_vu_1")
#games=("pokemon_red")

if [ -z "$model" ]; then
    echo "ERROR: 'model' variable is not set. Export model=<org/name> before running. Aborting."
    exit 1
fi

executors=("simple" "history" "sequence" "subgoal" "screendiff" "reflective" "spatialmap" "value" "belief" "adversarial" )

for game in "${games[@]}"; do
    echo "Running benchmark for game: $game"
    gen_model_save_name="${model#*/}"
    echo "  Running benchmark for model: $model"
    for executor in "${executors[@]}"; do
        echo "    Running benchmark for executor: $executor"
        bash scripts/benchmark.sh --game $game --executor $executor --executor_vlm_model "$model" --executor_vlm_kind vllm --max_steps 75 --regenerate true
    done
done