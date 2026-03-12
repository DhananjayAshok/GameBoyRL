source scripts/utils.sh
rm -rf cleanrl/videos/*
rm -rf cleanrl/runs/*
rm -rf cleanrl/wandb/*
python -c "from gameboy_worlds import clear_tmp_sessions; clear_tmp_sessions()"