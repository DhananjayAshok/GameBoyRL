"""
Writes a manifest that references several grouped-trajectory pickle files.

Each input pickle is a list[group], where each group is a list[trajectory], as written by
cleanrl/cleanrl_utils/group_trajectories.py. Instead of concatenating them (tens of GB of
frames), this writes a small manifest: a pickled list[str] of absolute paths to the input
pkls. Group order is positional, so downstream infer_tasks sees the same flattened sequence a
real concatenation would have produced. infer_tasks.load_grouped_trajectories understands
both a manifest (list[str]) and a plain list[group].

Used by scripts/pipeline/curiosity_all_tasks.sh to merge a game's per-init_state grouped
trajectories into a single file, so one infer_tasks call annotates all states.

The manifest points at the input pkls instead of copying them, so it breaks if those inputs
are later moved or deleted.

Output: <save_path>/grouped_<z_kind>_high_reward_trajectories.pkl — same naming as
group_trajectories.py so the combined file is interchangeable with a per-state one.

NOTE: curiosity_all_tasks.sh writes the combined file under an "all" init_state_group
directory, which would collide with a real init_state literally named "all".
"""
import os
import pickle

import click

from python_scripts.common import log


@click.command(name="combine_trajectories")
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
def combine_trajectories_cmd(input_paths, save_path, z_kind):
    """Write a manifest pkl referencing several grouped-trajectory pkls."""
    paths = [p.strip() for p in input_paths.split(",") if p.strip()]

    manifest = []
    for path in paths:
        if not os.path.exists(path):
            log(f"WARN: skipping missing grouped trajectory pkl: {path}")
            continue
        manifest.append(os.path.abspath(path))  # abspath so it resolves from any cwd

    if not manifest:
        log(
            f"WARN: no input paths exist across {len(paths)} path(s); writing empty manifest."
        )

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"grouped_{z_kind}_high_reward_trajectories.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(manifest, f)  # list[str] of input pkl paths, not the trajectories
    log(f"Wrote manifest of {len(manifest)} file(s) -> {out_path}")
