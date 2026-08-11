"""
Entry point for the benchmark_scripts package. Use --help for CLI options.

Three arms, one group: ``run_benchmark.py --game X baseline`` (no knowledge),
``... info`` (a hint from prebuilt documents) and ``... plan`` (a plan supervised to
completion). Called by scripts/benchmark.sh, scripts/benchmark_info.sh and run.sh.

Options shared by every arm live on the group and reach the subcommands through ctx.obj;
each arm declares only what is its own. That is what stops --max_replans being silently
accepted by the baseline, which a single flat command could not prevent. Group options must
precede the subcommand word.

Note for anyone resuming an old run: every arm's CSV carries a trailing session_dirs column
and no n_resets column, so CSVs written before this package cannot be resumed.
``load_checkpoint`` refuses them with an explanatory error rather than a pandas shape
error; the fix is --regenerate, or moving the old file aside.
"""

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
@click.option("--max_steps", default=200, type=int,
              help="Emulator steps for the WHOLE episode, every internal retry included.")
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
         save_video, max_steps, max_tool_calls, override_index, n_tasks,
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
