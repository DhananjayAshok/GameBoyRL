init_states=("default")

for init_state in "${init_states[@]}"; do
    bash scripts/train.sh --init_state "$init_state" --algorithm "random" --timesteps 50000 --max_steps 5000 --replay_buffer_save_folder randoms/$init_state
done