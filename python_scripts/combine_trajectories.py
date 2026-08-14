"""
Writes a manifest that references several grouped-trajectory pickle files.

Each input pickle is a list[group], where each group is a list[trajectory], as
written by cleanrl/cleanrl_utils/group_trajectories.py. Rather than loading and
concatenating all of them (tens of GB of frames) into one giant file, this writes
a small manifest: a pickled list[str] of absolute paths to the input pkls. Group
order is positional — first all groups of the first path, then the second, etc. —
so downstream infer_tasks sees the same flattened sequence a real concatenation
would have produced. infer_tasks.load_grouped_trajectories understands both a
manifest (list[str]) and a plain list[group], so a per-state file still works.

Used by scripts/pipeline/curiosity_all_tasks.sh to merge the per-init_state grouped
trajectories of a game into a single file, so one infer_tasks call annotates all
states. (infer_tasks keys its output only on run_name, with no init_state in the
path, so a per-init_state loop would let only the first state's annotation survive —
combining first is what makes every state contribute.)

The manifest points at the input pkls instead of copying them, so it breaks if
those inputs are later moved or deleted. The pipeline keeps them around, so this
is fine in practice.

Output: <save_path>/grouped_<z_kind>_high_reward_trajectories.pkl — same naming as
group_trajectories.py so the combined file is interchangeable with a per-state one.

NOTE: curiosity_all_tasks.sh writes the combined file under an "all" init_state_group
directory. If a game ever has a real init_state literally named "all", its grouped
trajectories directory collides with this combined directory — rename the combined
dir in curiosity_all_tasks.sh if that ever happens.
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
