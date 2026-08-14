"""
Small, cheap-to-import modules that both Python and Bash need.

Every module here must stay importable in a fraction of a second: ``python_funcs.py`` is
invoked once per lookup from shell scripts, so nothing in this package may import the VLM
stack at module scope. In practice that means importing only ``utils.fundamental``,
``utils.log_handling`` and ``utils.parameter_handling`` — never ``from utils import ...``,
which resolves lazily but still pulls whatever symbol you name.
"""
