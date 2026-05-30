# Clears transient CleanRL run artifacts (videos, runs, wandb logs) and purges
# temporary GameBoyWorlds emulator sessions. Safe to run at any time; does not
# touch stored models, replay buffers, or curiosity buffers.

source scripts/core/utils.sh
rm -rf cleanrl/videos/*
rm -rf cleanrl/runs/*
rm -rf cleanrl/wandb/*
python -c "from gameboy_worlds import clear_tmp_sessions; clear_tmp_sessions()"