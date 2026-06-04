source scripts/core/utils.sh
#games=("pokemon_red" "sword_of_hope_1" "sword_of_hope_2" "deja_vu_1" "deja_vu_2" "legend_of_zelda_links_awakening" "legend_of_zelda_the_oracle_of_seasons" "harvest_moon_1" "harvest_moon_2" "harvest_moon_3" "bomberman_pocket" "bomberman_quest" "bomberman_max")
games=("sword_of_hope_1" "sword_of_hope_2")
#games=("pokemon_red" "legend_of_zelda_links_awakening" "sword_of_hope_1" "harvest_moon_1" "bomberman_pocket" "deja_vu_1")
#games=("pokemon_red")

models=(
    "google/gemini-3.1-pro-preview"
)

#games=()
#models=(
#    "openai/gpt-4o-mini"
#)

executors=("reflective")

for game in "${games[@]}"; do
    echo "Running benchmark for game: $game"
    for model in "${models[@]}"; do
        gen_model_save_name="${model#*/}"
        echo "  Running benchmark for model: $model"
        for executor in "${executors[@]}"; do
            echo "    Running benchmark for executor: $executor"
            bash scripts/benchmark.sh --game $game --executor $executor --executor_vlm_model "$model" --executor_vlm_kind openrouter --max_steps 75
        done
    done
done