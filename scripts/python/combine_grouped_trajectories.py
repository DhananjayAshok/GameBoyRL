"""
Concatenates several grouped-trajectory pickle files into one.

Each input pickle is a list[group], where each group is a list[trajectory], as
written by cleanrl/cleanrl_utils/group_trajectories.py. Concatenation is a plain
list extend: the combined file is still a list[group] and group_idx is positional
across the union, so downstream infer_tasks consumes it unchanged.

Used by scripts/pipeline/curiosity_all_tasks.sh to merge the per-init_state grouped
trajectories of a game into a single file, so one infer_tasks call annotates all
states. (infer_tasks keys its output only on run_name, with no init_state in the
path, so a per-init_state loop would let only the first state's annotation survive —
combining first is what makes every state contribute.)

Output: <save_path>/grouped_<z_kind>_high_reward_trajectories.pkl — same naming as
group_trajectories.py so the combined file is interchangeable with a per-state one.

NOTE: curiosity_all_tasks.sh writes the combined file under an "all" init_state_group
directory. If a game ever has a real init_state literally named "all", its grouped
trajectories directory collides with this combined directory — rename the combined
dir in curiosity_all_tasks.sh if that ever happens.
"""
import os
import pickle
import sys

import click


def log(message):
    print(message, file=sys.stderr)


@click.command()
@click.option(
    "--input_paths",
    required=True,
    help="Comma-separated grouped-trajectory pkl paths to concatenate.",
)
@click.option(
    "--save_path",
    required=True,
    help="Directory to write the combined pkl into (created if absent).",
)
@click.option(
    "--z_kind",
    default="global",
    show_default=True,
    help="z_kind used in the output filename, matching group_trajectories.py.",
)
def combine(input_paths, save_path, z_kind):
    paths = [p.strip() for p in input_paths.split(",") if p.strip()]
    combined = []
    for path in paths:
        if not os.path.exists(path):
            log(f"WARN: skipping missing grouped trajectory pkl: {path}")
            continue
        with open(path, "rb") as f:
            groups = pickle.load(f)
        combined.extend(groups)

    if not combined:
        log(
            f"WARN: no groups found across {len(paths)} input path(s); writing empty combined file."
        )

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"grouped_{z_kind}_high_reward_trajectories.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(combined, f)
    log(f"Combined {len(combined)} groups from {len(paths)} file(s) -> {out_path}")


if __name__ == "__main__":
    combine()
