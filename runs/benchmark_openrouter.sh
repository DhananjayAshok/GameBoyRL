source scripts/core/utils.sh

games=(
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

models=(
    "google/gemini-3.6-flash"
    "anthropic/claude-haiku-4.5"
    "openai/gpt-5-mini"
)

executors=("single_visual")
supervisors=("subgoal")
max_steps=75
executor_max_new_tokens=16000
supervisor_max_new_tokens=10000

log_dir="logs/benchmark_openrouter"
mkdir -p "$log_dir"

for game in "${games[@]}"; do
    echo ""
    echo "=== game: $game ==="
    for executor in "${executors[@]}"; do
        for supervisor in "${supervisors[@]}"; do
            for model in "${models[@]}"; do
                log_file="$log_dir/${game}_$(model_save_name "$model")_${executor}_${supervisor}.out"
                echo "--- $game | $model | $executor | $supervisor -> $log_file"
                bash scripts/benchmark/run_benchmark.sh \
                    --game "$game" \
                    --supervisor "$supervisor" \
                    --executor "$executor" \
                    --executor_vlm_model "$model" \
                    --executor_vlm_kind openrouter \
                    --max_steps "$max_steps" \
                    --executor_max_new_tokens "$executor_max_new_tokens" \
                    --supervisor_max_new_tokens "$supervisor_max_new_tokens" \
                    > "$log_file" 2>&1 \
                    || echo "FAILED: $game | $model | $executor | $supervisor"
            done
        done
    done
done

echo "DONE ALL"
