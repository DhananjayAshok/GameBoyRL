bash scripts/default_rl.sh --algorithm sac --timesteps 1000000 --replay_buffer_save_folder metric_ablation/cosine/

bash scripts/default_rl.sh --algorithm sac --timesteps 1000000 --replay_buffer_save_folder metric_ablation/distance/ --similarity_metric distance

bash scripts/default_rl.sh --algorithm sac --timesteps 1000000 --replay_buffer_save_folder metric_ablation/hinge/ --similarity_metric hinge
