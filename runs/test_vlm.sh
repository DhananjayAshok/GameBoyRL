source scripts/utils.sh
run_name="sweep_attempt"
game="pokemon_red"


models=(
    "google/gemma-4-31b-it"
)

train_model="google/gemma-4-31b-it"

for model in "${models[@]}"; do
    gen_model_save_name="${model#*/}"
    model_save_name="${train_model#*/}"
    #bash scripts/create_dataset.sh --model_name "$model" --vlm_kind huggingface --game $game --init_state_group default --run_name $run_name

    #bash runs/benchmark.sh --game $game --executor_vlm_model "$model" --executor_vlm_kind huggingface

    #bash runs/benchmark.sh --game $game --executor_vlm_model "$train_model" --executor_vlm_kind huggingface --regenerate true --max_steps 75

    #bash scripts/train_vlm.sh --model_name "$train_model" --run_name $run_name --train_file $storage_dir/data/$game/$run_name/default/$gen_model_save_name/data.csv --overwrite true --push_to_hub y

    model_name="$huggingface_repo_namespace/$run_name-$model_save_name"

    bash runs/benchmark.sh --game $game --executor_vlm_model "$model_name" --executor_vlm_kind huggingface --regenerate true --max_steps 75
done
