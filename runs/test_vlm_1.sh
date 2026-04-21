source scripts/utils.sh
run_name="sweep_attempt"
game="pokemon_red"


models=(
    "Qwen/Qwen3-VL-32B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-2B-Instruct"
)
for model in "${models[@]}"; do
    model_save_name="${model#*/}"
    bash scripts/create_dataset.sh --model_name "$model" --vlm_kind huggingface --game $game --init_state_group default --run_name $run_name

    bash runs/benchmark.sh --game $game --executor_vlm_model "$model" --executor_vlm_kind huggingface

    bash scripts/train_vlm.sh --model_name "$model" --run_name $run_name --train_file $storage_dir/data/$game/$run_name/default/$model_save_name/data.csv --overwrite true --push_to_hub y

    model_name="$huggingface_repo_namespace/$run_name-$model_save_name"

    bash runs/benchmark.sh --game $game --executor_vlm_model "$model_name" --executor_vlm_kind huggingface
done