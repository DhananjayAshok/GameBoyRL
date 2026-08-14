"""
Shared helpers for the ``python_funcs.py`` command group.

Kept free of anything heavy — see the package docstring.
"""

import sys

import click


def log(message) -> None:
    """Diagnostic output for commands whose stdout is captured by Bash.

    Goes to **stderr**, always. A command invoked as ``$(python python_funcs.py ...)`` has its
    stdout read as the answer, so anything printed there that is not the answer corrupts it.
    """
    print(message, file=sys.stderr)


class MaybeNone(click.ParamType):
    """A string option where Bash's ``none`` sentinel means "not given".

    ``args_to_flags`` in ``scripts/core/utils.sh`` emits ``--key none`` for any empty value,
    and several defaults dicts carry a literal ``none`` (``output_dir``,
    ``supervisor_vlm_model``, ``buffer_load_path``, ...). Without this, a path command would
    take that at face value and build a path with a ``none`` segment in it, then hand it back
    with exit code 0 — a wrong answer rather than an error.
    """

    name = "text_or_none"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() == "none":
            return None
        return value


MAYBE_NONE = MaybeNone()
