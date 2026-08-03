"""
TEMPORARY entry point for the benchmark_scripts package. Use --help for CLI options.

This is the replacement for run_benchmark.py, run_benchmark_info.py and
run_benchmark_info_plan.py, living under a scratch name while the originals stay in place.
Nothing calls it yet: scripts/benchmark.sh, scripts/benchmark_info.sh and run.sh still
invoke the three original files, so a queued job that reaches its benchmark stage runs the
old code untouched.

At cutover this file becomes run_benchmark.py, the three originals are deleted, and those
three shell scripts gain the subcommand word (`run_benchmark.py baseline`, `... info`,
`... plan`).

Options shared by every arm live on the group and reach the subcommands through ctx.obj;
each arm declares only what is its own. That is what stops --max_replans being silently
accepted by the baseline, which a single flat command could not prevent.
"""

# TODO: CUTOVER — rename this file to run_benchmark.py and retire the originals.
#
# Nothing invokes this module yet. Until the steps below are done, the benchmark is still
# served by the three original files, so a queued job that reaches its benchmark stage runs
# the old code. Do all of it in one commit — a half-done cutover leaves run.sh calling a
# file that no longer takes the arguments it passes.
#
#   1. git mv run_benchmark_tmp.py run_benchmark.py     (overwrites the original)
#   2. git rm run_benchmark_info.py run_benchmark_info_plan.py
#   3. scripts/benchmark.sh       : python run_benchmark.py       -> ... run_benchmark.py baseline
#      scripts/benchmark_info.sh  : python run_benchmark_info.py  -> ... run_benchmark.py info
#                                   and rename its --hint_mode passthrough to --mode
#      run.sh                     : both invocations -> `run_benchmark.py baseline` and
#                                   `run_benchmark.py plan`
#   4. Group options must precede the subcommand: `run_benchmark.py --game X baseline`,
#      not `run_benchmark.py baseline --game X`. The shell scripts build one flag string,
#      so check where the subcommand word is spliced in.
#
# Two things that will bite, neither of which is a bug:
#
#   * --random_sample no longer exists (only --n_tasks). Nothing passes it today, but
#     grep before deleting: run.sh and scripts/info_plan_report.sh both build _firstN
#     suffixes from --n_tasks and would need matching edits if that ever changes.
#   * Every arm's CSV gained a trailing session_dirs column, so CSVs written by the old
#     runners cannot be resumed. load_checkpoint refuses them with an explanatory error
#     rather than a pandas shape error; the fix is --regenerate or moving the file aside.
#     Existing results/benchmark/**.csv are all old-schema.
#
# Verify after cutover with a real run before trusting it — see the module docstring in
# benchmark_scripts/common.py for what is shared and therefore what a mistake would break
# across all three arms at once.

import click

from gameboy_worlds import AVAILABLE_GAMES

from benchmark_scripts import baseline, info, plan
from execution.registry import AVAILABLE_EXECUTORS
from utils import load_parameters


@click.group()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--controller_variant", default="low_level", type=str)
@click.option("--executor", default="simple",
              type=click.Choice(list(AVAILABLE_EXECUTORS.keys())))
@click.option("--executor_vlm_model", default=None, type=str)
@click.option("--executor_vlm_kind", default=None, type=str)
@click.option("--save_video", type=bool, default=True)
@click.option("--max_resets", default=1, type=int)
@click.option("--max_steps", default=200, type=int,
              help="Emulator steps for the WHOLE episode, every attempt and retry included.")
@click.option("--max_tool_calls", default=0, type=int)
@click.option("--override_index", default=None, type=int, required=False)
@click.option("--n_tasks", type=int, default=None,
              help="Run the FIRST n tasks in benchmark order. A prefix rather than a "
                   "sample, so --n_tasks 5 is a subset of --n_tasks 10 and both are a "
                   "prefix of the full sweep. Omit to run every task.")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--regenerate", is_flag=True, default=False,
              help="Ignore any existing CSV for this run and start over.")
@click.pass_context
def main(ctx, game, controller_variant, executor, executor_vlm_model, executor_vlm_kind,
         save_video, max_resets, max_steps, max_tool_calls, override_index, n_tasks,
         verbose, regenerate):
    """Benchmark a frozen VLM on a game, with or without prebuilt knowledge."""
    parameters = load_parameters()
    vlm_name = executor_vlm_model or parameters["executor_vlm_model"]
    ctx.obj = dict(
        parameters=parameters,
        game=game,
        controller_variant=controller_variant,
        executor=executor,
        executor_vlm_model=executor_vlm_model,
        executor_vlm_kind=executor_vlm_kind,
        # Names the CSV and the emulator session directory. Derived once here so all three
        # arms agree on it — they write beside each other and a mismatch would be silent.
        model_save_name=vlm_name.split("/")[-1].lower(),
        save_video=save_video,
        max_resets=max_resets,
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        override_index=override_index,
        n_tasks=n_tasks,
        verbose=verbose,
        regenerate=regenerate,
    )


main.add_command(baseline, name="baseline")
main.add_command(info, name="info")
main.add_command(plan, name="plan")


if __name__ == "__main__":
    main()
