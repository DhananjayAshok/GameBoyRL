"""
Queries GameBoyWorlds for all available training states and writes them to
scripts/core/all_train_states.sh as a bash associative array (TRAIN_STATES).
Run this whenever the set of training states in GameBoyWorlds changes so that
create_all_traj.sh and propose_all_zeroshot.sh stay in sync.
"""
import os
from gameboy_worlds import get_all_training_states

# DELTA TODO: 1: Change this to print out only the init states from benchmark for those.
# That means each game dict will lead to a straight comma separated list of states.


if __name__ == "__main__":
    all_training_states = (
        get_all_training_states()
    )  # format {game: [state1, state2, ...]}

    output_path = os.path.join("scripts", "core", "all_train_states.sh")
    lines = [
        "# AUTO-GENERATED — do not edit by hand.",
        "# Maps (game, init_state_group) keys to comma-separated init_state names for all",
        "# supported games. Sourced by create_all_traj.sh and propose_all_zeroshot.sh to",
        "# iterate over every training init_state for a given game.",
        "# Regenerate by running: python scripts/python/create_task_dictionary.py",
        "",
        "declare -A TRAIN_STATES",
        "",
    ]

    for game, states_list in all_training_states.items():
        states_str = ",".join(states_list)
        lines.append(f'TRAIN_STATES["{game}"]="{states_str}"')

    lines += ["", "export TRAIN_STATES", ""]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved task dictionary to {output_path}")
