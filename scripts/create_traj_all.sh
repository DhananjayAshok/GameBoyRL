init_states=("default" "train_0" "train_1" "train_2" "train_3" "train_4")
init_state_group="default"
model_dir="iterative"
for init_state in "${init_states[@]}"; do
    if [[ "$init_state" == "${init_states[-1]}" ]]; then
        bash scripts/iterative_training.sh --init_state $init_state --sweep true --call_grouping true --init_state_group $init_state_group --model_dir $model_dir
    else
        bash scripts/iterative_training.sh --init_state $init_state --sweep true --call_grouping false --init_state_group $init_state_group --model_dir $model_dir
    fi
done