"""
Entry point for the benchmark_scripts package. Use --help for CLI options.

Five arms, one group, each adding one capability to the one before it::

    run_benchmark.py --game X baseline                  no supervisor reasoning at all
    run_benchmark.py --game X revision                  short legs, hint revised between them
    run_benchmark.py --game X subgoal                   a plan from the task alone
    run_benchmark.py --game X info_subgoal_parametric   planned from the model's own priors
    run_benchmark.py --game X info_subgoal_retrieval    planned from a distilled document

The subcommand word IS the supervisor's registry key (execution.registry.AVAILABLE_SUPERVISORS)
and the CSV stem, with the one exception that the control arm's key is ``dummy``. The two
info arms are separate commands rather than one command with a --mode flag, because they are
separate experiments with different inputs: --info_docs is required by one and rejected by
the other, which a shared command could not express.

Called by scripts/benchmark/run_benchmark.sh.

There used to be an ``info`` arm, which spent retrieved knowledge on a single hint written
once at the opening frame. It has been retired along with ``InfoHintSupervisor``, and its
absence is why knowledge is only available to an arm that plans: a document is spent
writing a plan, and without one there is nothing to spend it on.

Options shared by every arm live on the group and reach the subcommands through ctx.obj;
each arm declares only what is its own. That is what stops --max_replans being silently
accepted by the baseline, or --info_docs by an arm that has no plan to spend it on, which a
single flat command with a --supervisor flag could not prevent. Group options must precede
the subcommand word.

Note for anyone resuming an old run: every arm's CSV carries a trailing session_dirs column
and no n_resets column, so CSVs written before this package cannot be resumed.
``load_checkpoint`` refuses them with an explanatory error rather than a pandas shape
error; the fix is --regenerate, or moving the old file aside.
"""

import click

from gameboy_worlds import AVAILABLE_GAMES

from benchmark_scripts import (
    baseline,
    info_subgoal_parametric,
    info_subgoal_retrieval,
    revision,
    subgoal,
)
from execution.registry import AVAILABLE_EXECUTORS
from utils import load_parameters
from python_scripts.paths import model_save_name


@click.group()
@click.option("--game", default="pokemon_red", type=click.Choice(AVAILABLE_GAMES))
@click.option("--controller_variant", default="low_level", type=str)
@click.option("--executor", default="single_none",
              type=click.Choice(list(AVAILABLE_EXECUTORS.keys())))
@click.option("--executor_vlm_model", required=True, type=str,
              help="The model that plays. Required rather than falling back to the config's "
                   "executor_vlm_model: the supervisor's model defaults to THIS value, not to "
                   "the config, so an omitted flag left every supervised arm with no model at "
                   "all — and failed at the first supervisor call, after the emulator was up "
                   "and the CSV had been named from the config, which reads as a model "
                   "mismatch rather than a missing flag.")
@click.option("--executor_vlm_kind", required=True, type=str,
              help="VLM kind for --executor_vlm_model. Required for the same reason.")
@click.option("--supervisor_vlm_model", default=None, type=str,
              help="The one model every supervisor reasons with — selection, planning, "
                   "judging, hinting. Defaults to the executor's. A group option because "
                   "there are exactly two models anywhere in an arm, and this is the "
                   "second; the model that BUILT any knowledge read at test time is a "
                   "property of the document, recorded in its provenance, not a flag.")
@click.option("--supervisor_vlm_kind", default=None, type=str,
              help="VLM kind for --supervisor_vlm_model. Defaults to the executor's.")
@click.option("--supervisor_max_new_tokens", default=5000, show_default=True, type=int,
              help="Token budget for every supervisor call. One number rather than one per "
                   "stage: the stages used to differ and the only thing that bought was a "
                   "silent truncation when a reply outgrew its stage's allowance.")
@click.option("--save_video", type=bool, default=True)
@click.option("--max_steps", default=200, type=int,
              help="Emulator steps for the WHOLE episode, every internal retry included.")
@click.option("--max_tool_calls", default=0, type=int)
@click.option("--n_tasks", type=int, default=None,
              help="Run the FIRST n tasks in benchmark order. A prefix rather than a "
                   "sample, so --n_tasks 5 is a subset of --n_tasks 10 and both are a "
                   "prefix of the full sweep. Omit to run every task.")
@click.option("--verbose", is_flag=True, default=False)
@click.option("--regenerate", is_flag=True, default=False,
              help="Ignore any existing CSV for this run and start over.")
@click.option("--extra_name", default=None, type=str,
              help="Free-text discriminator appended to the CSV and session names, for runs "
                   "this identity cannot otherwise tell apart — two info_subgoal_retrieval "
                   "runs over different --docs_mode being the case it exists for, since which "
                   "documents were read reaches no other part of the name. Omit and the name "
                   "is unchanged. 'none' is accepted as the shell's absent sentinel.")
@click.option("--world_model_run_name", default=None, type=str,
              help="RL run_name whose trained world model and observation encoder drive "
                   "the 'world_model' executor. Required by that executor, ignored by the "
                   "others. 'none' is accepted as the shell's absent sentinel.")
@click.pass_context
def main(ctx, game, controller_variant, executor, executor_vlm_model, executor_vlm_kind,
         supervisor_vlm_model, supervisor_vlm_kind, supervisor_max_new_tokens,
         save_video, max_steps, max_tool_calls, n_tasks,
         verbose, regenerate, extra_name, world_model_run_name):
    """Benchmark a frozen VLM on a game, with or without prebuilt knowledge."""
    parameters = load_parameters()
    ctx.obj = dict(
        parameters=parameters,
        game=game,
        controller_variant=controller_variant,
        extra_name=extra_name,
        executor=executor,
        executor_vlm_model=executor_vlm_model,
        executor_vlm_kind=executor_vlm_kind,
        # Falls back to the executor's, which is now guaranteed to be a real model rather
        # than None. Resolved here rather than in each arm so the two arms cannot disagree
        # about what "unset" means.
        supervisor_vlm_model=supervisor_vlm_model or executor_vlm_model,
        supervisor_vlm_kind=supervisor_vlm_kind or executor_vlm_kind,
        supervisor_max_new_tokens=supervisor_max_new_tokens,
        # Names the CSV and the emulator session directory. Derived once here so both arms
        # agree on it — they write beside each other and a mismatch would be silent.
        model_save_name=model_save_name(executor_vlm_model),
        save_video=save_video,
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        n_tasks=n_tasks,
        verbose=verbose,
        regenerate=regenerate,
        # Spread into every arm's supervisor call, which forwards it verbatim to the
        # executor. Built here rather than per-arm so the four arms cannot disagree about
        # what the executor was given, and left empty unless actually set so a world-model
        # knob never lands in the init_kwargs of a run that has no world model.
        executor_kwargs=(
            {"world_model_run_name": world_model_run_name}
            if world_model_run_name and world_model_run_name != "none" else {}
        ),
    )


main.add_command(baseline, name="baseline")
main.add_command(revision, name="revision")
main.add_command(subgoal, name="subgoal")
main.add_command(info_subgoal_retrieval, name="info_subgoal_retrieval")
main.add_command(info_subgoal_parametric, name="info_subgoal_parametric")


if __name__ == "__main__":
    main()
