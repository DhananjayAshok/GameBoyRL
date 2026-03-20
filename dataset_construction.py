import json
import os
import pickle

import click
import pandas as pd
from PIL import Image

from task_annotation import high_level_action_to_string
from utils import load_parameters
from utils.vlm import convert_numpy_greyscale_to_pillow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_OPTIONS = "A, B, UP, LEFT, RIGHT, DOWN, START"

SPARSE_INPUT_TEMPLATE = (
    "You are playing {game}. Your task is: {task}. "
    "You see the current frame. What is your next action? "
    "Choose from: " + ACTION_OPTIONS
)

DENSE_INPUT_TEMPLATE = (
    "You are playing {game}. Your task is: {task}. "
    "You see the current frame. What is your next action and why? "
    "Choose from: " + ACTION_OPTIONS + ". "
    "Provide your reasoning then your action."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir(parameters: dict, game: str, trajectory_path: str, model_save_name: str) -> str:
    """
    Constructs the output directory:
      $data_dir / game / <relative path of pkl dir from storage_dir> / model_save_name
    """
    storage_dir = parameters["storage_dir"]
    data_dir = parameters["data_dir"]
    pkl_dir = os.path.dirname(os.path.abspath(trajectory_path))
    rel = os.path.relpath(pkl_dir, storage_dir)
    return os.path.join(data_dir, game, rel, model_save_name)


def _save_frame(frame, images_dir: str, group_idx, traj_idx: int, step: int) -> str:
    """Save a numpy frame as PNG, return its absolute path."""
    img: Image.Image = convert_numpy_greyscale_to_pillow(frame)
    filename = f"{group_idx}_{traj_idx}_{step}.png"
    path = os.path.join(images_dir, filename)
    img.save(path)
    return path


def _load_annotation(out_dir: str, prefix: str, model_save_name: str) -> dict:
    path = os.path.join(out_dir, f"{prefix}_{model_save_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Annotation not found at {path}. Run task_annotation.py first."
        )
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Click interface
# ---------------------------------------------------------------------------

@click.group()
@click.option("--game", required=True, help="Game name (e.g. 'Pokemon Red')")
@click.option("--trajectory_path", required=True, help="Path to grouped_high_reward_trajectories.pkl")
@click.option("--model_name", required=True, help="Model name used during annotation (e.g. gpt-4o)")
@click.option(
    "--max_rollback",
    default=None,
    type=int,
    show_default=True,
    help="Only use the last N steps of each trajectory. None = full trajectory.",
)
@click.pass_context
def main(ctx, game, trajectory_path, model_name, max_rollback):
    parameters = load_parameters()
    model_save_name = model_name.split("/")[-1]
    ctx.obj = dict(
        game=game,
        trajectory_path=trajectory_path,
        model_save_name=model_save_name,
        max_rollback=max_rollback,
        parameters=parameters,
    )


@main.command()
@click.pass_obj
def sparse(obj):
    """Build a sparse VLA dataset: one row per (frame, task paraphrase, action)."""
    game = obj["game"]
    trajectory_path = obj["trajectory_path"]
    model_save_name = obj["model_save_name"]
    max_rollback = obj["max_rollback"]
    parameters = obj["parameters"]

    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    pkl_dir = os.path.dirname(os.path.abspath(trajectory_path))
    task_annotation = _load_annotation(pkl_dir, "task_annotation", model_save_name)

    out_dir = _output_dir(parameters, game, trajectory_path, model_save_name)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    for group_idx, group in grouped_trajectories.items():
        group_key = str(group_idx)
        task_strings = task_annotation.get(group_key, task_annotation.get(group_idx))
        if not task_strings:
            print(f"Warning: no task strings for group {group_idx}, skipping.")
            continue

        for traj_idx, trajectory in enumerate(group):
            observations, actions, high_level_actions, rewards = trajectory

            # TODO: check obs shape — expected (num_frames, H, W, C) but may differ
            breakpoint()  # TODO: check obs shape before indexing
            n_steps = len(observations)
            start = 0 if max_rollback is None else max(0, n_steps - max_rollback)

            for step in range(start, n_steps - 1):
                frame = observations[step]
                image_path = _save_frame(frame, images_dir, group_idx, traj_idx, step)

                action_entry = high_level_actions[step] if step < len(high_level_actions) else (type(None), {})
                action_class, action_kwargs = action_entry
                action_str = high_level_action_to_string(action_class, action_kwargs)

                for task in task_strings:
                    rows.append({
                        "input": SPARSE_INPUT_TEMPLATE.format(game=game, task=task),
                        "image": image_path,
                        "output": action_str,
                    })

    df = pd.DataFrame(rows, columns=["input", "image", "output"])
    out_path = os.path.join(out_dir, "sparse_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved sparse dataset ({len(df)} rows) → {out_path}")


@main.command()
@click.pass_obj
def dense(obj):
    """Build a dense VLA dataset: one row per (frame, task paraphrase, reasoning + action)."""
    game = obj["game"]
    trajectory_path = obj["trajectory_path"]
    model_save_name = obj["model_save_name"]
    max_rollback = obj["max_rollback"]
    parameters = obj["parameters"]

    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    pkl_dir = os.path.dirname(os.path.abspath(trajectory_path))
    task_annotation = _load_annotation(pkl_dir, "task_annotation", model_save_name)
    dense_annotation = _load_annotation(pkl_dir, "dense_annotation", model_save_name)

    out_dir = _output_dir(parameters, game, trajectory_path, model_save_name)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    for group_idx, group in grouped_trajectories.items():
        group_key = str(group_idx)
        task_strings = task_annotation.get(group_key, task_annotation.get(group_idx))
        if not task_strings:
            print(f"Warning: no task strings for group {group_idx}, skipping.")
            continue

        records = dense_annotation.get(group_key, dense_annotation.get(group_idx, []))
        if not records:
            print(f"Warning: no dense annotation records for group {group_idx}, skipping.")
            continue

        for record in records:
            traj_idx = record["traj_idx"]
            step = record["step"]
            action_str = record["action_str"]
            reasoning = record.get("reasoning")

            # Apply max_rollback filter (records may cover more steps than desired)
            trajectory = group[traj_idx]
            observations = trajectory[0]
            # TODO: check obs shape — expected (num_frames, H, W, C) but may differ
            breakpoint()  # TODO: check obs shape before indexing
            n_steps = len(observations)
            start = 0 if max_rollback is None else max(0, n_steps - max_rollback)
            if step < start:
                continue

            frame = observations[step]
            image_path = _save_frame(frame, images_dir, group_idx, traj_idx, step)

            if reasoning is None:
                output_str = f"Action: {action_str}"
            else:
                output_str = f"Reasoning: {reasoning}\nAction: {action_str}"

            for task in task_strings:
                rows.append({
                    "input": DENSE_INPUT_TEMPLATE.format(game=game, task=task),
                    "image": image_path,
                    "output": output_str,
                })

    df = pd.DataFrame(rows, columns=["input", "image", "output"])
    out_path = os.path.join(out_dir, "dense_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved dense dataset ({len(df)} rows) → {out_path}")


if __name__ == "__main__":
    main()
