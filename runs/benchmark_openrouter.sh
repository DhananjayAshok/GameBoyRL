source scripts/core/utils.sh
games=("runes_of_virtue_1" "runes_of_virtue_2")
#games=("harvest_moon_1" "harvest_moon_2" "harvest_moon_3")
#games=("survival_kids_1")
#games=("pokemon_red" "legend_of_zelda_links_awakening" "sword_of_hope_1" "harvest_moon_1" "bomberman_pocket" "deja_vu_1")
#games=("pokemon_red")

models=(
    "google/gemini-3.1-pro-preview"
)

#games=()
#models=(
#    "openai/gpt-4o-mini"
#)

executors=("single_actions")

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