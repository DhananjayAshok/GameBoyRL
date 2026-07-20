"""
Read-only diagnostics for the GameBoyRL pipeline. Called by scripts/debug/*.sh.
Use --help for CLI options.

Each subcommand reads one stage's saved artifacts and writes a markdown report (plus any
figures it needs) to <results_dir>/debug/<game>/<stage>/. Nothing here trains, serves, or
queries a model, and nothing reads logs/ or slurm — only artifacts the pipeline is
guaranteed to have written. A missing artifact is a hard error naming the script that
produces it.

Group options identify *which experiment* is being inspected and must agree across every
report; per-command options control presentation (how many samples, how many frames) and
vary freely. --model_name is per-command because the curiosity stage's paths are not
keyed on a model.
"""

import click

from utils import load_parameters
from debug_scripts import (
    hello,
    debug_curiosity,
    debug_zeroshot,
    debug_attempt,
    debug_practice,
    debug_dataset,
    debug_benchmark,
)


@click.group()
@click.option("--game", required=True, help="Game name (e.g. deja_vu_1)")
@click.option("--run_name", default="my_run", show_default=True,
              help="Run name used by the RL/curiosity stages and the fine-tuned model name.")
@click.option("--executor", default="history", show_default=True,
              help="Executor short name; selects the attempts/practice dirs and benchmark CSVs.")
@click.option("--output_dir", default=None,
              help="Report root. Defaults to <results_dir>/debug/<game>.")
@click.option("--overwrite", is_flag=True, default=False,
              help="Re-render images that already exist instead of reusing them.")
@click.pass_context
def main(ctx, game, run_name, executor, output_dir, overwrite):
    parameters = load_parameters()
    ctx.obj = dict(
        game=game,
        run_name=run_name,
        executor=executor,
        output_dir=output_dir,
        overwrite=overwrite,
        parameters=parameters,
    )


main.add_command(hello, name="hello")
main.add_command(debug_curiosity, name="curiosity")
main.add_command(debug_zeroshot, name="zeroshot")
main.add_command(debug_attempt, name="attempt")
main.add_command(debug_practice, name="practice")
main.add_command(debug_dataset, name="dataset")
main.add_command(debug_benchmark, name="benchmark")


if __name__ == "__main__":
    main()
