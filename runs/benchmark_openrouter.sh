source scripts/utils.sh
#games=("pokemon_red" "sword_of_hope_1" "sword_of_hope_2" "deja_vu_1" "deja_vu_2" "legend_of_zelda_links_awakening" "legend_of_zelda_the_oracle_of_seasons" "harvest_moon_1" "harvest_moon_2" "harvest_moon_3" "bomberman_pocket" "bomberman_quest" "bomberman_max")
games=("pokemon_red" "legend_of_zelda_links_awakening" "legend_of_zelda_the_oracle_of_seasons" "sword_of_hope_1" "sword_of_hope_2" "harvest_moon_1" "harvest_moon_2" "harvest_moon_3" "bomberman_pocket" "bomberman_quest" "bomberman_max" "deja_vu_1" "deja_vu_2")

models=(
    "google/gemini-3.1-pro-preview"
    "openai/gpt-4o"
    "anthropic/claude-opus-4.7"
    "qwen/qwen3-vl-235b-a22b-instruct"
    "google/gemma-4-31b-it"
)

#games=()
#models=(
#    "openai/gpt-4o-mini"
#)

for game in "${games[@]}"; do
    echo "Running benchmark for game: $game"
    for model in "${models[@]}"; do
        gen_model_save_name="${model#*/}"
        echo "  Running benchmark for model: $model"
        bash runs/benchmark.sh --game $game --executor_vlm_model "$model" --executor_vlm_kind openrouter --max_steps 75
    done
done