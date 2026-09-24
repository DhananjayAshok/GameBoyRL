"""
The single entry point Bash uses to ask Python for anything it used to derive itself.

    python python_funcs.py <command> [--flag value ...]

Most commands print exactly one line to stdout — the answer — and nothing else. Bash captures
that with ``$(...)``, so **stdout is a data channel**: every diagnostic in every command
reached from here must go to stderr (``python_scripts.common.log``, or ``utils.log_handling``).
A stray ``print`` here silently corrupts a path at the call site.

Use ``path_of`` in ``scripts/core/utils.sh`` rather than calling this directly: ``$(...)``
swallows a non-zero exit, so an unchecked capture turns a failed lookup into an empty string.

Kept cheap to import — ~0.1s, against ~15s if anything touches torch. Nothing on the import
path may pull in the VLM stack; the commands that need heavy imports do them inside the
function body.

Project parameters are loaded LAZILY, through ``ctx.obj["parameters"]()``.
"""

import functools

import click

from python_scripts.combine_trajectories import combine_trajectories_cmd
from python_scripts.embed_markdown import embed_markdown_cmd
from python_scripts.paths_cli import PATH_COMMANDS
from python_scripts.strings import strings_cmd
from python_scripts.task_dictionary import task_dictionary_cmd


@click.group()
@click.pass_context
def main(ctx):
    """Path lookups and small helpers shared by the Python and Bash halves of the pipeline."""
    from utils.parameter_handling import load_parameters

    # A callable, not the dict: commands that do not need parameters do not pay for them.
    ctx.obj = {"parameters": functools.cache(load_parameters)}


main.add_command(strings_cmd)
main.add_command(task_dictionary_cmd)
main.add_command(combine_trajectories_cmd)
main.add_command(embed_markdown_cmd)
for _command in PATH_COMMANDS:
    main.add_command(_command)


if __name__ == "__main__":
    main()
