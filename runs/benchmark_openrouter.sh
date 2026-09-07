source scripts/core/utils.sh

games=(
    "bomberman_max"
    "bomberman_pocket"
    "bomberman_quest"
    "deja_vu_1"
    "deja_vu_2"
    "harvest_moon_1"
    "harvest_moon_2"
    "harvest_moon_3"
    "legend_of_zelda_the_oracle_of_seasons"
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

for game in "${games[@]}"; do
    echo ""
    echo "=== game: $game ==="
    for model in "${models[@]}"; do
        for executor in "${executors[@]}"; do
            for supervisor in "${supervisors[@]}"; do
                echo "--- $game | $model | $executor | $supervisor"
                bash scripts/benchmark/run_benchmark.sh \
                    --game "$game" \
                    --supervisor "$supervisor" \
                    --executor "$executor" \
                    --executor_vlm_model "$model" \
                    --executor_vlm_kind openrouter \
                    --max_steps "$max_steps" \
                    --executor_max_new_tokens "$executor_max_new_tokens" \
                    --supervisor_max_new_tokens "$supervisor_max_new_tokens" \
                    || echo "FAILED: $game | $model | $executor | $supervisor"
            done
        done
    done
done

echo "DONE ALL"
