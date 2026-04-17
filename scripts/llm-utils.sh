#!/usr/bin/env bash

# Capture the full command the user wants to run
if [ $# -eq 0 ]; then
    echo "Usage: bash llm-utils.sh <command_to_run>"
    exit 1
fi

# Save the full command (all args)
to_run_command="$@"

currdir="$PWD"
source scripts/utils.sh || { echo "Could not source utils.sh"; exit 1; }
python configs/create_env_file.py || { echo "Could not create env file"; exit 1; }

# Enter llm-utils and activate its environment
source "llm-utils/setup/.venv/bin/activate" || { echo "Could not source llm-utils env"; exit 1; }

# Go back into llm-utils where the command should run
cd llm-utils || { echo "Could not cd into llm-utils"; exit 1; }

# Run the full command exactly as provided
echo "Running command: $to_run_command"
#eval "$to_run_command"


# Return and reactivate project environment
cd "$currdir" || { echo "Could not return to original directory"; exit 1; }
source "$env_dir/bin/activate" || { echo "Could not source project env"; exit 1; }