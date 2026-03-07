from poke_worlds import (
    AVAILABLE_GAMES,
    get_environment,
    get_benchmark_tasks,
    get_test_environment,
)
import click
from utils import load_parameters
from tqdm import tqdm
import pandas as pd
import traceback


def run_task(row, max_resets, controller_variant, **emulator_kwargs):
    success = False
    n_resets = 1
    n_steps_total = 0
    n_steps = 0
    mission = row["task"]
    task_str = mission.replace(" ", "_").lower()
    emulator_kwargs = emulator_kwargs.copy()
    emulator_kwargs["session_name"] += f"/{task_str}/"
    environment = get_test_environment(
        row=row, controller_variant=controller_variant, **emulator_kwargs
    )
    try:
        while n_resets < max_resets + 1:
            environment.reset()
            rewards, terminated, truncated = environment.human_step_play(
                max_steps=emulator_kwargs.get("max_steps", 1000), show_info=True
            )
            if terminated:
                success = True
                n_steps = len(rewards)
                break
            else:
                n_resets += 1
                n_steps += len(rewards)

    except Exception as e:
        traceback.print_exc()
        print(f"Error during execution of task '{mission}': {e}")
    environment.close()
    return success, n_resets - 1, n_steps


@click.command()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option(
    "--controller_variant",
    default="state_wise",
    type=str,
)
@click.option("--save_video", type=bool, default=True)
@click.option("--max_resets", default=3, type=int)
@click.option("--max_steps", default=1000, type=int)
@click.option("--override_index", default=None, type=int, required=False)
def do(game, controller_variant, save_video, max_resets, max_steps, override_index):
    project_parameters = load_parameters()
    executor_vlm_name = project_parameters["executor_vlm_model"]
    model_save_name = executor_vlm_name.split("/")[-1].lower()
    session_name = f"benchmark_zero_shot_{model_save_name}"
    headless = True
    emulator_kwargs = {
        "headless": headless,
        "save_video": save_video,
        "session_name": session_name,
        "max_steps": max_steps,
    }
    benchmark_tasks = get_benchmark_tasks(game=game)
    results = []
    columns = ["game", "task", "success", "n_resets", "n_steps"]
    for i, row in tqdm(benchmark_tasks.iterrows(), total=len(benchmark_tasks)):
        if override_index is not None and i != override_index:
            continue
        if override_index is not None:
            print(f"Running override index {override_index} on row:")
            for column in row.index:
                print(f"  {column}: {row[column]}")
        success, n_resets, n_steps = run_task(
            row=row,
            max_resets=max_resets,
            controller_variant=controller_variant,
            **emulator_kwargs,
        )
        results.append([row["game"], row["task"], success, n_resets, n_steps])
        df = pd.DataFrame(results, columns=columns)
        save_path = f"benchmark_zero_shot_{game}_{model_save_name}.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved benchmark results to {save_path}")


if __name__ == "__main__":
    do()
