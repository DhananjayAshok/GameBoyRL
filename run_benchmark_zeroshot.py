"""
Usage: python run_benchmark_zeroshot.py --game pokemon_red --save_video True --max_resets 3 --max_steps 30
python run_benchmark_zeroshot.py --game pokemon_red --save_video True --max_resets 3 --max_steps 30 --random_sample 2 --verbose
"""

from gameboy_worlds import (
    AVAILABLE_GAMES,
    get_environment,
    get_benchmark_tasks,
    get_test_environment,
)
import click
from utils import load_parameters
from execution.registry import AVAILABLE_EXECUTORS
from deep_test_executor import print_trajectory
from tqdm import tqdm
import pandas as pd
import traceback
import os


def run_task(row, max_resets, controller_variant, executor_class, max_tool_calls, vlm_model=None, vlm_kind=None, verbose=False, **emulator_kwargs):
    success = False
    n_resets = 1
    n_steps_total = 0
    n_invalid_total = 0
    subgoals_reached = []
    subgoals_all = None
    mission = row["task"]
    task_str = mission.replace(" ", "_").lower()
    emulator_kwargs = emulator_kwargs.copy()
    emulator_kwargs["session_name"] += f"/{task_str}/"
    try:
        while n_resets < max_resets + 1:
            environment = get_test_environment(
                row=row, controller_variant=controller_variant, **emulator_kwargs
            )
            executor = executor_class(
                env=environment,
                task=mission,
                max_steps=emulator_kwargs["max_steps"],
                max_tool_calls=max_tool_calls,
                vlm_model=vlm_model,
                vlm_kind=vlm_kind,
            )
            report = executor.report
            last_state = environment.get_info()

            if subgoals_all is None:
                subgoals_all = last_state["subgoals"]["all"]
            for subgoal in last_state["subgoals"]["completed"]:
                if subgoal not in subgoals_reached:
                    subgoals_reached.append(subgoal)

            n_steps_total += last_state["core"]["steps"]
            n_invalid_total += len(report.invalid_steps)

            if verbose:
                print(f"\n  Reset {n_resets} trajectory:")
                print_trajectory(report, name=row["game"] + "/" + emulator_kwargs["session_name"])

            environment.close()

            if report.termination_reason == "terminated":
                success = True
                break
            n_resets += 1

    except Exception as e:
        traceback.print_exc()
        print(f"Error during execution of task '{mission}': {e}")
    return success, n_resets - 1, n_steps_total, n_invalid_total, subgoals_reached, subgoals_all


@click.command()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--controller_variant", default="low_level", type=str)
@click.option("--executor", default="simple", type=click.Choice(list(AVAILABLE_EXECUTORS.keys())))
@click.option("--executor_vlm_model", default=None, type=str)
@click.option("--executor_vlm_kind", default=None, type=str)
@click.option("--save_video", type=bool, default=True)
@click.option("--max_resets", default=3, type=int)
@click.option("--max_steps", default=200, type=int)
@click.option("--max_tool_calls", default=0, type=int)
@click.option("--override_index", default=None, type=int, required=False)
@click.option("--random_sample", type=int, default=None)
@click.option("--verbose", is_flag=True, default=False)
@click.option("--regenerate", is_flag=True, default=False)
def do(
    game,
    controller_variant,
    executor,
    executor_vlm_model,
    executor_vlm_kind,
    save_video,
    max_resets,
    max_steps,
    max_tool_calls,
    override_index,
    random_sample,
    verbose,
    regenerate,
):
    project_parameters = load_parameters()
    vlm_name = executor_vlm_model or project_parameters["executor_vlm_model"]
    model_save_name = vlm_name.split("/")[-1].lower()
    session_name = f"benchmark_zero_shot_{executor}_{model_save_name}"
    headless = True
    emulator_kwargs = {
        "headless": headless,
        "save_video": save_video,
        "session_name": session_name,
        "max_steps": max_steps,
    }
    benchmark_tasks = get_benchmark_tasks(game=game)
    results = []
    columns = [
        "game",
        "task",
        "success",
        "n_resets",
        "n_steps",
        "n_invalid",
        "subgoals_reached",
        "all_subgoals",
    ]
    if random_sample is not None:
        if not (1 <= random_sample <= len(benchmark_tasks) - 1):
            raise ValueError(
                f"random_sample must be between 1 and {len(benchmark_tasks) - 1}, got {random_sample}"
            )
        benchmark_tasks = benchmark_tasks.sample(
            n=random_sample, random_state=42
        ).reset_index(drop=True)
    os.makedirs(f"results/{game}/", exist_ok=True)
    if random_sample is not None:
        save_path = f"results/{game}/{executor}_{model_save_name}_sample{random_sample}.csv"
    else:
        save_path = f"results/{game}/{executor}_{model_save_name}.csv"
    if not regenerate and os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        results = existing_df.values.tolist()
        n_completed = len(existing_df)
        print(f"Resuming from checkpoint: {n_completed} tasks already completed in {save_path}")
    else:
        results = []
        n_completed = 0
    for i, row in tqdm(benchmark_tasks.iterrows(), total=len(benchmark_tasks)):
        if i < n_completed:
            continue
        if override_index is not None and i != override_index:
            continue
        if override_index is not None:
            print(f"Running override index {override_index} on row:")
            for column in row.index:
                print(f"  {column}: {row[column]}")
        success, n_resets, n_steps, n_invalid, subgoals_reached, subgoals_all = run_task(
            row=row,
            max_resets=max_resets,
            controller_variant=controller_variant,
            executor_class=AVAILABLE_EXECUTORS[executor],
            max_tool_calls=max_tool_calls,
            vlm_model=executor_vlm_model,
            vlm_kind=executor_vlm_kind,
            verbose=verbose,
            **emulator_kwargs,
        )
        results.append(
            [
                row["game"],
                row["task"],
                success,
                n_resets,
                n_steps,
                n_invalid,
                subgoals_reached,
                subgoals_all,
            ]
        )
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(save_path, index=False)
        print(f"Saved benchmark results to {save_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    do()
