import click
import os
from utils import load_parameters, log_info, log_warn, log_error
from PIL import Image
import json
import pandas as pd
import pickle
from tqdm import tqdm
from utils.vlm import convert_numpy_greyscale_to_pillow
from execution.executor import SimpleExecutor

def get_output(reasoning, action):
    return f"Reasoning: {reasoning}\nAction: {action}"


def get_input(game, task):
    action_list = """
    Arrow Keys (UP for up, DOWN for down, LEFT for left, RIGHT for right), A and B for buttons, START for start.
    """
    prompt = SimpleExecutor.STEP_PROMPT
    prompt = prompt.replace("[GAME]", game)
    prompt = prompt.replace("[TASK]", task)
    prompt = prompt.replace("[ACTION_LIST]", action_list)
    prompt = prompt.replace("[TOOLS_BLOCK]", "")
    prompt = prompt.replace("[ERROR_BLOCK]", "")
    prompt = prompt.replace("[TOOL_RESULT_BLOCK]", "")
    prompt = prompt.replace("[ACTION_FORMAT]", "ACTION: <action>")
    return prompt


@click.command()
@click.option("--push_to_hub", is_flag=True, help="Whether to push the dataset to Hugging Face Hub")
@click.pass_obj
def save(obj, push_to_hub):
    """Organize trajectory annotations into a single file and image folder for training."""
    parameters = load_parameters()
    trajectory_path = obj["trajectory_path"]
    game = obj["game"]
    model_name = obj["model_name"]
    model_save_name = model_name.split("/")[-1]
    out_dir = os.path.dirname(trajectory_path)
    out_path = os.path.join(out_dir, f"dense_annotation_{model_save_name}.json")
    if not os.path.exists(trajectory_path):
        log_error(f"Trajectory path {trajectory_path} does not exist.", parameters=parameters)    
        
    data_dir = parameters["data_dir"] 
    rest_of_path = out_dir.split("grouped_trajectories/")[-1]
    data_dir = os.path.join(data_dir, rest_of_path, model_save_name)
    image_dir = os.path.join(data_dir, "images")
    if os.path.exists(data_dir):
        if obj["overwrite"]:
            log_warn(f"{data_dir} already exists. Overwriting...")
        else:
            log_error(f"{data_dir} already exists. Use --overwrite to overwrite.", parameters=parameters)
    os.makedirs(image_dir, exist_ok=True)
    df_out_path = os.path.join(data_dir, "data.csv")
    if os.path.exists(df_out_path):
        if obj["overwrite"]:
            log_warn(f"{df_out_path} already exists. Overwriting...")
        else:
            log_error(f"{df_out_path} already exists. Use --overwrite to overwrite.", parameters=parameters)
    log_info(f"Saving annotations from {out_path} and saving to {df_out_path}. Images will be saved to {image_dir}", parameters=parameters)
    columns = ["group_idx", "traj_idx", "step_idx", "task", "input", "image", "reasoning", "action", "output"]
    data = []
    with open(out_path, "r") as f:
        annotations = json.load(f)
    with open(trajectory_path, "rb") as f:
        grouped_trajectories = pickle.load(f)
    
    group_idxes = annotations.keys()
    for group_idx in tqdm(group_idxes, desc="Processing groups", total=len(group_idxes)):
        all_steps = annotations[group_idx] # list
        for step in tqdm(all_steps, desc="Processing steps", total=len(all_steps), leave=False):
            if not step['good_action']:
                continue
            traj_idx = step["traj_idx"]
            observations, actions, high_level_actions, rewards = grouped_trajectories[int(group_idx)][traj_idx]
            step_idx = step["step"]
            image = observations[step_idx]
            task = step["task_name"]
            reasoning = step["reasoning"]
            action = step["action_str"]            
            image_name = f"group_{group_idx}_traj_{traj_idx}_step_{step_idx}.png"
            image_path = os.path.abspath(os.path.join(image_dir, image_name))
            img = convert_numpy_greyscale_to_pillow(image)
            img.save(image_path)
            input_text = get_input(game, task)
            output = get_output(reasoning, action)
            data.append([group_idx, traj_idx, step_idx, task, input_text, image_path, reasoning, action, output])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(df_out_path, index=False)
    log_info(f"Saved dataset with {len(data)} rows to {df_out_path}", parameters=parameters)
    if push_to_hub:
        # TODO
        pass

@click.command()
@click.pass_obj
def load(obj):
    """
    Load the dataset from the expected huggingface hub path 
    """
    # TODO
    pass
        