source scripts/core/utils.sh
max_steps=20
games=("survival_kids_1")

models=(
    "gpt-4o-mini"
)

executors=("history")

for game in "${games[@]}"; do
    echo "Running benchmark for game: $game"
    for model in "${models[@]}"; do
        gen_model_save_name="${model#*/}"
        echo "  Running benchmark for model: $model"
        for executor in "${executors[@]}"; do
            echo "    Running benchmark for executor: $executor"
            bash scripts/benchmark.sh --game $game --executor $executor --executor_vlm_model "$model" --executor_vlm_kind openrouter --max_steps $max_steps --regenerate true
        done
    done
done