# trajectories are saved in parameters["storage_dir"]/grouped_trajectories/$game/<path>/grouped_high_reward_trajectories.pkl
# it is a dictionary with numeric keys and values that are lists of trajectories
# each trajectory is a list (observations, actions, high_level_actions, rewards)
# observations is a stack of frames (numpy array) with shape (num_frames, 144, 160, 1)
# actions is a list of integers with shape (num_frames-1,)
# high_level_actions is a list of integers with shape (num_frames-1,)
# rewards is a list of floats with shape (num_frames-1,)
# the grouping is done by similarity of the final frames of the trajectories

import json
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
import click
from PIL import Image, ImageDraw, ImageFont

from utils import load_parameters
from utils.vlm import VLM, convert_numpy_greyscale_to_pillow


def plot_obs_transitions(observations, save_name):
    # given list of frames (H x W) plot all the frames with horizontal concatenation as an image
    total_height = observations[0].shape[0]
    total_width = sum(obs.shape[1] for obs in observations)
    single_image = np.zeros((total_height, total_width), dtype=observations[0].dtype)
    current_x = 0
    for i, obs in enumerate(observations):
        w = obs.shape[1]
        single_image[:, current_x:current_x+w] = obs[:, :]  # assuming (H, W, 1)
        current_x += w
    img = convert_numpy_greyscale_to_pillow(single_image)
    draw = ImageDraw.Draw(img)
    current_x = 0
    for i, obs in enumerate(observations):
        w = obs.shape[1]
        bg = np.mean(obs[:8, :max(w, 8)])
        fill = 0 if bg > 127 else 255
        draw.text((current_x + 2, 2), str(i), fill=fill)
        current_x += w
    img.save(save_name)    



@click.command()
@click.option("--trajectory_path", required=True, help="Path to grouped_high_reward_trajectories.pkl")
@click.option("--frac", default=1.0, show_default=True, help="Fraction of observation transitions to sample")
@click.option("--group_idx", default=None, type=int, help="If specified, only plot this group.")
@click.option("--traj_idx", default=None, type=int, help="If specified, only plot this trajectory (requires --group_idx).")
def show(trajectory_path, frac, group_idx, traj_idx):
    """Randomly sample a fraction of observation transitions and save them as images."""
    assert traj_idx is None or group_idx is not None, "--traj_idx requires --group_idx to be specified."
    parameters = load_parameters()
    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)

    os.makedirs(parameters["tmp_dir"], exist_ok=True)

    if group_idx is not None:
        indexes = [group_idx]
    else:
        indexes = list(range(len(grouped_trajectories)))
        random.shuffle(indexes)
        indexes = indexes[:int(len(grouped_trajectories) * frac)]

    for i, trajectory_group in tqdm(enumerate(grouped_trajectories), desc="Plotting observation transitions", total=len(grouped_trajectories)):
        if i not in indexes:
            continue
        group_dir = parameters["tmp_dir"] + f"/trajectories/group_{i}"
        os.makedirs(group_dir, exist_ok=True)
        if traj_idx is not None:
            trajectory = trajectory_group[traj_idx]
            observations = trajectory[0]
            save_path = group_dir + f"/traj_{traj_idx}.png"
            plot_obs_transitions(observations, save_path)
            print(f"Saved {save_path}")
        else:
            for j, trajectory in tqdm(enumerate(trajectory_group), leave=False, total=len(trajectory_group)):
                observations = trajectory[0]
                plot_obs_transitions(observations, group_dir + f"/traj_{j}.png")
            if group_idx is not None:
                print(f"Saved {group_dir}/")
    if group_idx is None and traj_idx is None:
        print(f"Saved {parameters['tmp_dir']}/trajectories/")


if __name__ == "__main__":
    show()
