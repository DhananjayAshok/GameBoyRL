source scripts/utils.sh
run_name="sweep_attempt"
games=("pokemon_red" "sword_of_hope_1" "sword_of_hope_2" "deja_vu_1" "deja_vu_2" "legend_of_zelda_links_awakening" "legend_of_zelda_the_oracle_of_seasons")


models=(
    "google/gemini-3.1-flash-lite-preview"
    "google/gemini-3.1-pro-preview"
    "openai/gpt-4o"
    "anthropic/claude-4-sonnet"
    "anthropic/claude-4-opus"
)

for game in "${games[@]}"; do
    echo "Running benchmark for game: $game"
    for model in "${models[@]}"; do
        gen_model_save_name="${model#*/}"
        echo "  Running benchmark for model: $model"
        bash runs/benchmark.sh --game $game --executor_vlm_model "$model" --executor_vlm_kind openrouter
    done
done