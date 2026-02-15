source setup/.venv/bin/activate
source configs/config.env


cd cleanrl

# RL Arguments
algo="sac"

# Environment Arguments
max_steps=3000
game="pokemon_red"
controller="low_level"
train_env_name="default"
test_env_name="default"
train_init_state="none"
test_init_state="none"

# Curiosity Arguments
observation_embedder="random_patch"
reset_curiosity_module="true"
if [ "$reset_curiosity_module" = "true" ]; then
    argpart="--reset-curiosity-module"
else
    argpart=""
fi
curiosity_module="embedbuffer"


train_env_id="poke_worlds-$game-$train_env_name-$train_init_state-$controller-$max_steps-False"
test_env_id="poke_worlds-$game-$test_env_name-$test_init_state-$controller-$max_steps-True"
exp_name="$algo-$observation_embedder-$curiosity_module-$reset_curiosity_module-$train_env_id"
model_save_path="$storage_dir/models/$exp_name/"


echo "Starting Experiment: $exp_name"
#python cleanrl/sac_atari.py --seed 1 --env-id $train_env_id --total-timesteps 3000000 --track --wandb-project-name $WANDB_PROJECT --model_save_path $storage_dir/models/$train_env_id/ --capture_video --save_model &> ../$train_env_id.out
python cleanrl/${algo}_curiosity.py --exp_name $exp_name --seed 1 --env-id $train_env_id --total-timesteps 3000000 --track \
    --wandb-project-name $WANDB_PROJECT --model_save_path $model_save_path --capture_video --save_model \
    --observation-embedder $observation_embedder $argpart \
    --curiosity-module $curiosity_module &> ../$exp_name.out


python cleanrl_utils/enjoy.py --exp-name sac_atari --model_path $model_save_path/model.pt \
    --env-id $test_env_id --save-name $exp_name