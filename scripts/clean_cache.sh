source configs/config.env
rm -rf cleanrl/videos/*
rm -rf cleanrl/runs/*
rm -rf cleanrl/wandb/*
python -c "from poke_worlds import clear_tmp_sessions; clear_tmp_sessions()"