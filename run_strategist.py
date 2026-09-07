"""
Entry point for a strategist run. Use --help for CLI options.

Not a benchmark arm. Every arm in ``run_benchmark.py`` sweeps a fixed list of short tasks
and scores each independently; a strategist run is the opposite shape — one long-horizon
goal, one emulator session, and a sequence of tasks the strategist chooses as it goes. There
is no task list to sweep and no per-task pass rate to average, so it gets its own command
rather than a sixth subcommand whose options would mean something different from the other
five.

    run_strategist.py --game pokemon_red --goal "Obtain your first Pokemon" \\
        --executor_vlm_model MODEL --executor_vlm_kind vllm

Writes a JSON record of the episode under ``storage_dir/strategist/``, plus the emulator's
own session directory (video, frames) when --save_video is on.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import click

from gameboy_worlds import AVAILABLE_GAMES, get_benchmark_tasks, get_test_environment

from execution.registry import AVAILABLE_EXECUTORS
from execution.strategists import AVAILABLE_STRATEGISTS
from python_scripts import paths
from utils import load_parameters, log_info

#: Where a strategist PLAYTHROUGH begins, per game, and which tracker scores it.
#:
#: **The state must match the goal.** This is the single most expensive thing to get wrong
#: here and it fails silently, twice over in this project's history:
#:
#: - `default` on pokemon_red is Viridian City with no starter, several screens from Oak's
#:   lab. Six runs were spent there on "obtain your first Pokemon". Nothing errored; the
#:   reflection call simply hallucinated a professor onto an empty street.
#: - `starter` puts the agent INSIDE Oak's Lab beside the Poke Balls. The milestone is then
#:   a few button presses away, so the agent never walks downstairs, never leaves the house
#:   and never meets Oak — it measures the last ten seconds of the opening, not a playthrough.
#:
#: Verify a start state by SCREENSHOT before running anything against it.
GAME_START = {
    # `initial` is the TRUE start of the game: Player's House Second Floor, the bedroom,
    # `current_starter: None`. Verified by screenshot.
    #
    # It replaces `starter`, which put the agent INSIDE Oak's Lab beside the three Poke
    # Balls. That made "obtain a Pokemon" reachable in a few presses and so made the
    # milestone cheap: the agent never walked downstairs, never left the house, never met
    # Oak. A playthrough has to start where the player starts.
    #
    # `collect_championship_test` rather than a plain tracker: it exposes all eight badges
    # as SUBGOALS and terminates only on beating the Champion, so progress is ground truth
    # from the environment instead of a number this code scrapes out of RAM itself.
    "pokemon_red": {"init_state": "initial",
                    "state_tracker_class": "collect_championship_test"},
    "pokemon_crystal": {"init_state": "initial",
                        "state_tracker_class": "collect_championship_test"},
    # Prism and Brown use the init_state their own playthrough benchmark row uses, so a
    # run here is comparable to that row.
    "pokemon_prism": {"init_state": "post_intro",
                      "state_tracker_class": "collect_championship_test"},
    "pokemon_brown": {"init_state": "location_merson_city",
                      "state_tracker_class": "collect_championship_test"},
}


@click.command()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--goal", default=None, type=str,
              help="The long-horizon goal, in the words the planner should reason about. "
                   "Omit when --benchmark_task is given: the task's own text becomes the goal.")
@click.option("--benchmark_task", default=None, type=int,
              help="Index into the game's benchmark tasks. Switches the run to MEASURED mode: "
                   "the task text becomes the goal, the episode runs in the `test` "
                   "environment, and that task's tracker decides success instead of a model. "
                   "This is the only mode whose control-vs-ablation numbers are comparable "
                   "across arms, and the only one on the same scale as run_benchmark.py. "
                   "Use with --strategist tracker.")
@click.option("--strategist", default="notebook",
              type=click.Choice(list(AVAILABLE_STRATEGISTS.keys())))
@click.option("--executor", default="single_actions",
              type=click.Choice(list(AVAILABLE_EXECUTORS.keys())),
              help="Defaults to single_actions rather than single_none: a strategist plans "
                   "on top of what the executor did, and an executor with no action history "
                   "cannot report that coherently.")
@click.option("--controller_variant", default="low_level", type=str)
@click.option("--executor_vlm_model", required=True, type=str, help="The model that plays.")
@click.option("--executor_vlm_kind", required=True, type=str)
@click.option("--strategist_vlm_model", default=None, type=str,
              help="The model that plans. Defaults to the executor's.")
@click.option("--strategist_vlm_kind", default=None, type=str)
@click.option("--max_tasks", default=8, show_default=True, type=int,
              help="Planned tasks before giving up. Each is a whole supervised episode, so "
                   "this multiplies the cost of everything else.")
@click.option("--max_steps_per_task", default=175, show_default=True, type=int)
@click.option("--max_tool_calls", default=0, show_default=True, type=int)
@click.option("--strategist_max_new_tokens", default=1200, show_default=True, type=int)
@click.option("--use_notebook", default=True, show_default=True, type=bool,
              help="False runs the ablation: the planner sees only the previous attempt "
                   "instead of the accumulated notebook.")
@click.option("--verify_goal", default=True, show_default=True, type=bool,
              help="Spend a probe task confirming the goal on screen. False makes the run "
                   "cheaper and its success flag untrustworthy.")
@click.option("--save_video", default=True, show_default=True, type=bool)
@click.option("--extra_name", default=None, type=str,
              help="Free-text discriminator for the output name.")
@click.option("--stop_on_goal", default=True, show_default=True, type=bool,
              help="Allow the run to end early when the goal looks reached. Set FALSE for a "
                   "progression goal ('play as far as you can'), which has no completion "
                   "condition: the ledger will answer yes to the first sign of progress and "
                   "end a whole-game run after one task.")
@click.option("--done_check_every_k", default=None, type=int,
              help="Run the executor's completion check every k-th step instead of every "
                   "step. LOOKS like a cheap 40% saving and is usually a trap: the done "
                   "check is the ONLY channel the executor has for saying 'this task is "
                   "finished'. Between checks a model that believes it is done emits an "
                   "empty Action, which scores as INVALID, and four in a row kill the leg "
                   "(MAX_CONSECUTIVE_INVALID=4). A run at k=5 died with tasks lasting 26 "
                   "seconds each. Safe only where tasks are never satisfied early. Leave "
                   "unset for anything compared against run_benchmark.py numbers.")
@click.option("--goal_ledger_key", default="has_pokemon", show_default=True, type=str,
              help="Ledger key whose truthiness means the goal is reached. Empty string "
                   "disables the ledger heuristic, which is what a progression goal wants: "
                   "'play as far as you can' has no completion condition and the ledger will "
                   "answer yes to the first sign of progress.")
@click.option("--verbose", is_flag=True, default=False)
def main(game, goal, benchmark_task, strategist, executor, controller_variant, executor_vlm_model,
         executor_vlm_kind, strategist_vlm_model, strategist_vlm_kind, max_tasks,
         max_steps_per_task, max_tool_calls, strategist_max_new_tokens, use_notebook,
         verify_goal, stop_on_goal, save_video, extra_name, done_check_every_k, verbose):
    """Pursue one long-horizon goal with a strategist."""
    parameters = load_parameters()
    if game not in GAME_START:
        raise click.ClickException(
            f"No start state configured for {game}. Add one to GAME_START in "
            f"run_strategist.py — a strategist starts where the game starts, and there is "
            f"no benchmark row to take that from."
        )

    strategist_class = AVAILABLE_STRATEGISTS[strategist]
    executor_class = AVAILABLE_EXECUTORS[executor]
    if done_check_every_k:
        # A named subclass rather than mutating the shared class attribute: the registry's
        # classes are global, and changing one in place would silently alter every other
        # arm in the same process. Named so the identity survives into the record.
        executor_class = type(
            f"{executor_class.__name__}_k{done_check_every_k}",
            (executor_class,),
            {"DONE_CHECK_EVERY_K_STEPS": done_check_every_k,
             "__doc__": f"{executor} with the completion check every {done_check_every_k} steps."},
        )
    model_save_name = paths.model_save_name(executor_vlm_model)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{strategist}_{executor}_{model_save_name}"
    if extra_name and extra_name != "none":
        run_name += f"_{extra_name}"
    if not use_notebook:
        # In the name, not just the record: the ablation writes beside the control and a
        # pair of files that differ only inside would be indistinguishable in a directory
        # listing, which is how an ablation gets mistaken for its own control.
        run_name += "_nonotebook"

    # --- measured mode: a benchmark task, graded by its tracker -------------------------
    if benchmark_task is not None:
        tasks = get_benchmark_tasks(game=game)
        if not 0 <= benchmark_task < len(tasks):
            raise click.ClickException(
                f"--benchmark_task {benchmark_task} out of range: {game} has {len(tasks)} tasks."
            )
        row = tasks.iloc[benchmark_task].to_dict()
        goal = goal or row["task"]
        if strategist != "tracker":
            raise click.ClickException(
                "--benchmark_task requires --strategist tracker. The other strategists judge "
                "the goal with a model, which would throw away the tracker verdict that "
                "makes this mode measurable."
            )
        run_name += f"_task{benchmark_task}"
        environment = get_test_environment(
            row=row,
            controller_variant=controller_variant,
            headless=True,
            save_video=save_video,
            session_name=f"strategist/{game}/{run_name}/{stamp}/",
            max_steps=max_tasks * max_steps_per_task + 1000,
            wait_ticks=20,
        )
        log_info(f"MEASURED mode: benchmark task {benchmark_task} "
                 f"({row['state_tracker_class']}) decides success", parameters)

    # --- open-goal mode: no tracker exists, so a model judges the goal -------------------
    elif goal is None:
        raise click.ClickException("--goal is required unless --benchmark_task is given.")

    else:
        start = GAME_START[game]
        # The TEST environment, not "default". A playthrough is scored on the eight badge
        # SUBGOALS that `collect_championship_test` exposes, and only a TestTracker
        # surfaces `subgoals` in get_info(). The earlier objection to test environments --
        # that they truncate once their one task is unwinnable -- does not apply here:
        # PokemonRedChampionshipTerminateMetric defines termination only (beat the
        # Champion) and never truncates, so a long run is safe in it.
        environment = get_test_environment(
            row={"game": game, **start},
            controller_variant=controller_variant,
            headless=True,
            save_video=save_video,
            session_name=f"strategist/{game}/{run_name}/{stamp}/",
            max_steps=max_tasks * max_steps_per_task + 1000,
            wait_ticks=20,
        )
        log_info(f"PLAYTHROUGH: {game} from '{start['init_state']}' "
                 f"scored by {start['state_tracker_class']}", parameters)

    log_info(f"Strategist run: {run_name} on {game}", parameters)
    log_info(f'Goal: "{goal}"', parameters)

    try:
        agent = strategist_class(
            goal=goal,
            game=game,
            env=environment,
            executor_class=executor_class,
            vlm_model=strategist_vlm_model or executor_vlm_model,
            vlm_kind=strategist_vlm_kind or executor_vlm_kind,
            executor_vlm_model=executor_vlm_model,
            executor_vlm_kind=executor_vlm_kind,
            max_tasks=max_tasks,
            max_steps_per_task=max_steps_per_task,
            max_tool_calls=max_tool_calls,
            max_new_tokens=strategist_max_new_tokens,
            use_notebook=use_notebook,
            verify_goal=verify_goal,
            stop_on_goal=stop_on_goal,
            # Only in measured mode. A `test` environment truncates once its task can no
            # longer be won, so without a reset every attempt after the first runs against a
            # dead episode. An open-goal run must NOT reset: progress through the world is
            # the thing being accumulated.
            reset_between_tasks=benchmark_task is not None,
            goal_ledger_key=goal_ledger_key,
            parameters=parameters,
            verbose=verbose,
            on_task_complete=_checkpoint,
        )
        report = agent.run()
    finally:
        # Always: an emulator left open holds its session directory and its video handle, and
        # a crashed run that keeps them is a run whose partial video cannot be read.
        environment.close()

    with open(out_path, "w") as handle:
        json.dump(_serialise(report), handle, indent=2)

    print()
    print(str(report))
    print()
    print(f"Record written to {out_path}")


def _serialise(report) -> dict:
    """
    The episode as JSON-safe data.

    Reports are kept as their rendered strings rather than as structures: the supervisor
    report's own ``__str__`` is what every existing reader of this pipeline consumes, and
    duplicating its fields here would create a second format to keep in step with it.
    """
    return {
        "goal": report.goal,
        "game": report.game,
        "strategist": report.strategist_name,
        "init_kwargs": report.init_kwargs,
        "goal_achieved": report.goal_achieved,
        "stop_reason": report.stop_reason,
        "n_planned_tasks": len(report.planned_tasks),
        "n_successful_tasks": report.n_successful_tasks,
        "n_steps": report.n_steps,
        "n_invalid": report.n_invalid,
        "strategist_input_tokens": report.strategist_input_tokens,
        "strategist_output_tokens": report.strategist_output_tokens,
        "notebook": report.notebook,
        "tasks": [
            {
                "index": task.index,
                "task": task.task,
                "hint": task.hint,
                "success": task.success,
                "lesson": task.lesson,
                "is_verification": task.is_verification,
                "progress": task.progress,
                # Explicitly, not buried in the report string: "terminated" is the
                # environment's tracker firing (ground truth) while "agent_done" is the
                # executor grading itself. Those mean very different things and the
                # difference was only recoverable by inference before this was recorded.
                "termination_reasons": (
                    [leg.termination_reason for leg in task.report.executor_reports]
                    if task.report is not None else []
                ),
                "report": str(task.report) if task.report is not None else None,
            }
            for task in report.tasks
        ],
        "vlm_calls": [
            {
                "stage": call.stage,
                "prompt": call.prompt,
                "response": call.response,
                "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens,
            }
            for call in report.vlm_calls
        ],
    }


if __name__ == "__main__":
    main()
