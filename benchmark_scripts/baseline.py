"""
The no-knowledge arm: run the executor on each benchmark task with nothing but the task
string.

This is the control every other arm is measured against, which is why it takes no options
of its own beyond the shared ones — anything that would make it cleverer would stop it
being a baseline.
"""

from __future__ import annotations

import json

import click

from gameboy_worlds import get_benchmark_tasks

from execution.registry import AVAILABLE_EXECUTORS

from benchmark_scripts import common


@click.command(name="baseline")
@click.pass_obj
def baseline_cmd(obj):
    """Benchmark a frozen model with no hints, plans or retrieved knowledge."""
    parameters = obj["parameters"]
    game = obj["game"]
    executor = obj["executor"]
    executor_class = AVAILABLE_EXECUTORS[executor]
    model_save_name = obj["model_save_name"]

    emulator_kwargs = {
        "headless": True,
        "save_video": obj["save_video"],
        "session_name": f"benchmark_zero_shot_{executor}_{model_save_name}",
        "max_steps": obj["max_steps"],
    }

    columns = common.COMMON_COLUMNS + [common.SESSION_COLUMN]
    tasks = common.select_tasks(get_benchmark_tasks(game=game), obj["n_tasks"])
    save_path = common.results_path(parameters, game,
                                    f"{executor}_{model_save_name}", obj["n_tasks"])
    results, n_completed = common.load_checkpoint(save_path, obj["regenerate"],
                                                  columns, parameters)

    def run_one(row):
        def play(environment, reset_idx):
            # Constructing an Executor runs the episode; `report` is populated by the time
            # the constructor returns.
            executor_instance = executor_class(
                env=environment,
                task=row["task"],
                game=row["game"],
                max_steps=obj["max_steps"],
                max_tool_calls=obj["max_tool_calls"],
                vlm_model=obj["executor_vlm_model"],
                vlm_kind=obj["executor_vlm_kind"],
            )
            report = executor_instance.report
            if obj["verbose"]:
                print(f"\n  Reset {reset_idx} trajectory:")
                report.show()
            return common.PlayResult(
                report=report,
                report_str=str(report),
                n_invalid=len(report.invalid_steps),
                legs=[{"label": "episode", "call_log": report.vlm_call_log}],
            )

        return common.run_episode(
            row, play,
            arm="baseline",
            max_resets=obj["max_resets"],
            controller_variant=obj["controller_variant"],
            executor_name=executor_class.__name__,
            model=model_save_name,
            **emulator_kwargs,
        )

    def build_row(row, outcome):
        return common.common_row(row, outcome) + [json.dumps(outcome.session_dirs)]

    common.run_sweep(
        tasks,
        columns=columns,
        save_path=save_path,
        results=results,
        n_completed=n_completed,
        override_index=obj["override_index"],
        run_one=run_one,
        build_row=build_row,
    )
