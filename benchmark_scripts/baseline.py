"""
The no-knowledge arm: run the executor on each benchmark task with nothing but the task
string.
"""

from __future__ import annotations

import json

import click

from gameboy_worlds import get_benchmark_tasks

from execution.registry import AVAILABLE_EXECUTORS, AVAILABLE_SUPERVISORS
from execution.supervisors import DummySupervisor

from benchmark_scripts import common
from python_scripts import paths


@click.command(name="baseline")
@click.pass_obj
def baseline_cmd(obj):
    """Benchmark a frozen model with no hints, plans or retrieved knowledge."""
    parameters = obj["parameters"]
    game = obj["game"]
    executor = obj["executor"]
    controller_variant = obj["controller_variant"]
    extra_name = obj["extra_name"]
    executor_class = AVAILABLE_EXECUTORS[executor]
    model_save_name = obj["model_save_name"]
    supervisor_name = "dummy"
    supervisor_class = AVAILABLE_SUPERVISORS[supervisor_name]

    emulator_kwargs = {
        "headless": True,
        "save_video": obj["save_video"],
        "session_name": paths.benchmark_session_name(supervisor=supervisor_name, executor=executor, controller_variant=controller_variant,
                                                     model=model_save_name,
                                                     extra_name=extra_name),
        "max_steps": obj["max_steps"],
    }

    columns = common.COMMON_COLUMNS + [common.SESSION_COLUMN]
    tasks = common.select_tasks(get_benchmark_tasks(game=game), obj["n_tasks"])
    save_path = common.results_path(parameters, game, supervisor=supervisor_name, executor=executor, controller_variant=controller_variant,
                                    model=model_save_name, extra_name=extra_name,
                                    n_tasks=obj["n_tasks"])
    results, n_completed = common.load_checkpoint(save_path, obj["regenerate"],
                                                  columns, parameters)

    def run_one(row):
        def play(environment):
            # Through DummySupervisor rather than straight to the executor: every arm
            # produces a SupervisorReport, so the archive and every reader have one shape.
            # This supervisor adds no reasoning, which is what makes it the control.
            supervisor = supervisor_class(
                task=row["task"],
                executor_class=executor_class,
                env=environment,
                game=row["game"],
                max_steps=obj["max_steps"],
                max_tool_calls=obj["max_tool_calls"],
                # Passed like every other arm. It happens to match `load_parameters()`
                # today, which is why omitting it was harmless, but an arm that overrides a
                # parameter in-process (info_subgoal does, for the executor token budget)
                # would otherwise be running against a different dict from this one.
                parameters=parameters,
                vlm_model=obj["executor_vlm_model"],
                vlm_kind=obj["executor_vlm_kind"],
                **obj["executor_kwargs"],
            )
            result = supervisor.evaluate()
            if obj["verbose"]:
                for leg in result["report"].executor_reports:
                    leg.show()
            return common.PlayResult(report=result["report"])

        return common.run_episode(
            row, play,
            supervisor=supervisor_name,
            controller_variant=obj["controller_variant"],
            executor_name=executor_class.__name__,
            model=model_save_name,
            extra_name=extra_name,
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
        run_one=run_one,
        build_row=build_row,
    )
