source scripts/core/utils.sh
max_steps=2
games=("harry_potter_philosophers_stone" "harry_potter_chamber_of_secrets")

models=(
    "gpt-4o-mini"
)

executors=("single_actions")

for game in "${games[@]}"; do
    echo "Running benchmark for game: $game"
    for model in "${models[@]}"; do
        gen_model_save_name="${model#*/}"
        echo "  Running benchmark for model: $model"
        for executor in "${executors[@]}"; do
            echo "    Running benchmark for executor: $executor"
            bash scripts/benchmark/run_benchmark.sh --game $game --executor $executor --executor_vlm_model "$model" --executor_vlm_kind openrouter --max_steps $max_steps --regenerate true
        done
    done
done