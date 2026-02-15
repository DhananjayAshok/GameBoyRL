source setup/.venv/bin/activate
source configs/config.env


cd cleanrl

# Environment Arguments
max_steps=3000
game="pokemon_red"
controller="low_level"
train_env_name="default"
test_env_name="default"
train_init_state="default"
test_init_state="default"

# Curiosity Arguments
observation_embedder="random_patch"
reset_curiosity_module="true"
curiosity_module="embedbuffer"


train_env_id="poke_worlds-$game-$env_name-$train_init_state-$controller-$max_steps-False"
test_env_id="poke_worlds-$game-$env_name-$test_init_state-$controller-$max_steps-True"

#python cleanrl/sac_atari.py --seed 1 --env-id $train_env_id --total-timesteps 3000000 --track --wandb-project-name $WANDB_PROJECT --model_save_path $storage_dir/models/$train_env_id/ --capture_video --save_model &> ../$train_env_id.out
python cleanrl/sac_curiosity.py --seed 1 --env-id $train_env_id --total-timesteps 3000000 --track \
    --wandb-project-name $WANDB_PROJECT --model_save_path $storage_dir/models/$train_env_id/ --capture_video --save_model \
    --observation-embedder $observation_embedder --reset-curiosity-module $reset_curiosity_module \
    --curiosity-module $curiosity_module &> ../$train_env_id.out




python cleanrl_utils/enjoy.py --exp-name sac_atari --model_path $storage_dir/models/$env_id/model.pt \
    --env-id $train_env_id --save-name TRAIN-$train_env_id-TEST-$test_env_id