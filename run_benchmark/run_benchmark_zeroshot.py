from gameboy_worlds import (
    AVAILABLE_GAMES,
    get_environment,
    get_benchmark_tasks,
    get_test_environment,
)
import click
from utils import load_parameters
from execution.supervisor import SimpleSupervisor
from execution.pokemon.executors import PokemonExecutor
from execution.pokemon.reports import SimplePokemonExecutionReport
from tqdm import tqdm
import pandas as pd
import traceback
import os


def run_task(row, max_resets, controller_variant, **emulator_kwargs):
    success = False
    n_resets = 1
    n_steps_total = 0
    n_steps = 0
    subgoals_reached = []
    subgoals_all = None
    mission = row["task"]
    task_str = mission.replace(" ", "_").lower()
    emulator_kwargs = emulator_kwargs.copy()
    emulator_kwargs["session_name"] += f"/{task_str}/"
    environment = get_test_environment(
        row=row, controller_variant=controller_variant, **emulator_kwargs
    )
    supervisor = SimpleSupervisor(
        game=row["game"],
        environment=environment,
        executor_class=PokemonExecutor,
        execution_report_class=SimplePokemonExecutionReport,
    )
    supervisor.setup_play(
        mission=mission,
        initial_visual_context="You are seeing a screenshot of the game.",
    )
    try:
        while n_resets < max_resets + 1:
            supervisor_report = supervisor.play()
            if len(supervisor_report.execution_reports) > 0:
                last_kwargs, last_execution_report = (
                    supervisor_report.execution_reports[-1]
                )
                last_execution_states = last_execution_report.get_state_infos()
                # updated n_steps
                if len(last_execution_states) == 0:  # manually check for success
                    last_state = environment.get_info()
                else:
                    last_state = last_execution_states[-1]
                # update subgoals
                if subgoals_all is None:
                    subgoals_all = last_state["subgoals"]["all"]
                subgoals_completed = last_state["subgoals"]["completed"]
                for subgoal in subgoals_completed:
                    if subgoal not in subgoals_reached:
                        subgoals_reached.append(subgoal)
                step_count = last_state["core"]["steps"]
                n_steps = step_count  # this counts all steps across resets
                n_steps_total += n_steps
                if last_execution_report.exit_code == 2:
                    success = True
                    break
                if "termination_truncation" in last_state:
                    if last_state["termination_truncation"]["terminated"]:
                        success = True
                        break
            n_steps = 0
            n_resets += 1

    except Exception as e:
        traceback.print_exc()
        print(f"Error during execution of task '{mission}': {e}")
    environment.close()
    return success, n_resets - 1, n_steps_total, subgoals_reached, subgoals_all


@click.command()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option(
    "--controller_variant",
    default="state_wise",
    type=str,
)
@click.option("--save_video", type=bool, default=True)
@click.option("--max_resets", default=3, type=int)
@click.option("--max_steps", default=200, type=int)
@click.option("--override_index", default=None, type=int, required=False)
@click.option("--random_sample", type=int, default=None)
def do(
    game,
    controller_variant,
    save_video,
    max_resets,
    max_steps,
    override_index,
    random_sample,
):
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
    columns = [
        "game",
        "task",
        "success",
        "n_resets",
        "n_steps",
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
    os.makedirs("results", exist_ok=True)
    for i, row in tqdm(benchmark_tasks.iterrows(), total=len(benchmark_tasks)):
        if override_index is not None and i != override_index:
            continue
        if override_index is not None:
            print(f"Running override index {override_index} on row:")
            for column in row.index:
                print(f"  {column}: {row[column]}")
        success, n_resets, n_steps, subgoals_reached, subgoals_all = run_task(
            row=row,
            max_resets=max_resets,
            controller_variant=controller_variant,
            **emulator_kwargs,
        )
        results.append(
            [
                row["game"],
                row["task"],
                success,
                n_resets,
                n_steps,
                subgoals_reached,
                subgoals_all,
            ]
        )
        df = pd.DataFrame(results, columns=columns)
        save_path = f"results/benchmark_zero_shot_{game}_{model_save_name}.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved benchmark results to {save_path}")


if __name__ == "__main__":
    do()
