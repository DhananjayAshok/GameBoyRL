source scripts/utils.sh || { echo "Could not source utils"; exit 1; }
init_state_group="default"
game="pokemon_red"
init_states="default,train_0,train_1,train_2,train_3,train_4"
run_name="default"
z_min=6

bash scripts/create_traj.sh --game "$game" --init_state_group "$init_state_group" --init_states "$init_states" --run_name "$run_name"


bash scripts/group_trajectories.sh --game ${ARGS["game"]} --replay_buffer_folder $run_name/ --save_path $storage_dir/grouped_trajectories/$game/$run_name/ --z_min $z_min

python show_trajectories.py --name pokemon_red_default --trajectory_path $storage_dir/grouped_trajectories/$game/$run_name/grouped_global_high_reward_trajectories.pkl

sbatch rl