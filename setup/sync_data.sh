source scripts/core/utils.sh || { echo "Could not source utils"; exit 1; }
python -m gameboy_worlds.setup_data pull --game all