"""
Structured strings — RL experiment names and env IDs — built from --key value argument pairs.

Called by bash through ``python_funcs.py strings ...`` (via utils.sh get_string_from_args).
Extensible: add a new StringFunction subclass and register it in STRING_FUNCTIONS to expose a
new string kind. Prints exactly one line to stdout; all diagnostics go to stderr.

Does NOT load project parameters: an experiment name is a pure function of its arguments, and
these are called in RL script hot paths.
"""

from abc import ABC, abstractmethod

import click

from python_scripts.common import log  # noqa: F401  (kept on the module's surface)


def depathify_legacy(string) -> str:
    """The WEAK depathify: folds only ``/``, ``\\`` and space. Used by :class:`ExperimentName` only.

    Everything else uses ``utils.fundamental.depathify``, which folds every non-word
    character. Switching :class:`ExperimentName` onto the strong one would rename the run
    directories under ``<storage>/models/`` and ``<storage>/logs/``, and reaches cleanrl's
    ``--exp_name``.

    :param string: the string to depathify
    :type string: str
    :return: the depathified string
    :rtype: str
    """
    return string.replace("/", "_").replace("\\", "_").replace(" ", "_")


class StringFunction(ABC):
    NAME = None  # name of the string function, used to call it from bash.
    # By convention, all args are case insensitive and are not allowed to use ' '
    REQUIRED_ARGS = (
        []
    )  # list of required arguments that the function needs to run. If any of these are missing, the function will raise an error.
    OPTIONAL_ARGS = (
        {}
    )  # dict of optional arguments that the function can take, with the key as the argument name and the value as the default value. If any of these are missing, the function will use the default value. If any unexpected arguments are passed, they will be ignored.
    # You typically do NOT want to be using optional args. Instead, make a common set of optional args in the bash utils.sh file and source that in your bash script. This way, you can easily update the optional args without having to change the python code.

    def __init__(self):
        if self.NAME is None:
            raise ValueError("StringFunction must have a NAME attribute")
        if " " in self.NAME:
            raise ValueError(
                f"StringFunction NAME cannot contain spaces. Got: {self.NAME}"
            )
        for i in range(len(self.REQUIRED_ARGS)):
            self.REQUIRED_ARGS[i] = self.REQUIRED_ARGS[i].lower()
            if " " in self.REQUIRED_ARGS[i]:
                raise ValueError(
                    f"Argument names cannot contain spaces. Got: {self.REQUIRED_ARGS[i]}"
                )
        for arg in self.OPTIONAL_ARGS:
            arg = arg.lower()
            if " " in arg:
                raise ValueError(f"Argument names cannot contain spaces. Got: {arg}")
            if arg in self.REQUIRED_ARGS:
                raise ValueError(
                    f"Argument {arg} cannot be both required and optional."
                )

    def validate_args(self, **kwargs):
        for arg in self.REQUIRED_ARGS:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")
        for arg in kwargs:
            if arg not in self.REQUIRED_ARGS and arg not in self.OPTIONAL_ARGS:
                pass  # unexpected arguments are allowed, just ignored.

    @abstractmethod
    def _get_string(self, **kwargs) -> str:
        """
        Kwargs is guaranteed to have all keys filled.
        """
        pass  # write the logic to return the string you want here.

    def get_string(self, **kwargs) -> None:
        self.validate_args(**kwargs)
        for arg, default_value in self.OPTIONAL_ARGS.items():
            if arg not in kwargs:
                kwargs[arg] = default_value
        string = self._get_string(**kwargs)
        print(
            string
        )  # print the string to stdout, which will be captured by the bash script


# Implement a function below and add it to the STRING_FUNCTIONS list to make it available for use in bash.


class ExperimentName(StringFunction):
    NAME = "exp_name"
    REQUIRED_ARGS = [
        "game",
        "init_state",
        "max_steps",
        "controller",
        "env",
        "algorithm",
        "timesteps",
        "gamma",
        "observation_embedder",
        "embedder_load_path",
        "similarity_metric",
        "curiosity_module",
        "ocr_alpha",
        "buffer_load_path",
        "seed",
    ]

    def add_to_exp_name(self, exp_name, keys, kwargs):
        for i, key in enumerate(keys):
            exp_name += f"{kwargs[key]}"
            if i != len(keys) - 1:
                exp_name += "_"
        return exp_name

    def _get_string(self, **kwargs) -> str:
        exp_name = ""
        exp_name = self.add_to_exp_name(
            exp_name,
            [
                "game",
                "init_state",
                "max_steps",
                "controller",
                "env",
                "algorithm",
                "timesteps",
                "gamma",
                "observation_embedder",
            ],
            kwargs,
        )
        if kwargs["embedder_load_path"] is not None:
            exp_name += f"-{depathify_legacy(kwargs['embedder_load_path'])}_"
        else:
            exp_name += f"_"
        exp_name = self.add_to_exp_name(
            exp_name,
            keys=["similarity_metric", "curiosity_module", "ocr_alpha"],
            kwargs=kwargs,
        )
        if kwargs["buffer_load_path"] is not None:
            exp_name += f"-{depathify_legacy(kwargs['buffer_load_path'])}_"
        else:
            exp_name += f"_"
        exp_name += f"{kwargs['seed']}"
        return exp_name


class EnvID(StringFunction):
    NAME = "env_id"
    REQUIRED_ARGS = ["game", "env", "init_state", "controller", "max_steps"]

    def _get_string(self, **kwargs) -> str:
        return f"gameboy_worlds-{kwargs['game']}-{kwargs['env']}-{kwargs['init_state']}-{kwargs['controller']}-{kwargs['max_steps']}"


STRING_FUNCTIONS = [ExperimentName, EnvID]

ALL_STRING_FUNCTIONS = {func.NAME.lower(): func() for func in STRING_FUNCTIONS}


def parse_pairs(args) -> dict:
    """Turn a flat ``["--key", "value", ...]`` list into a dict, mapping ``none`` -> ``None``.

    Not click-parsed: the keys are whatever ``args_to_flags`` happens to emit for the caller's
    ARGS dict, which varies per script, so there is no fixed option set to declare.
    """
    args = list(args)
    # must be an even number, matching --key value pairs
    if len(args) % 2 != 0:
        raise ValueError(f"Arguments must be in the format --key value. Got: {args}")
    arg_dict = {}
    for i in range(0, len(args), 2):
        if not args[i].startswith("--"):
            raise ValueError(f"Argument keys must start with --. Got: {args[i]}")
        if args[i].strip("--") in arg_dict:
            raise ValueError(f"Duplicate argument key: {args[i][2:]}")
        if args[i + 1].startswith("--"):
            raise ValueError(f"consecutive --s: {args}")
        if args[i + 1].strip().lower() == "none":
            arg_dict[args[i].strip("--").lower()] = None
        else:
            arg_dict[args[i].strip("--").lower()] = args[i + 1]
    return arg_dict


@click.command(
    name="strings",
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
)
@click.argument("string_kind")
@click.pass_context
def strings_cmd(ctx, string_kind):
    """Print one structured string (exp_name, env_id) built from --key value pairs.

    Everything after STRING_KIND is forwarded verbatim, so the caller's ARGS dict can be
    splatted in with args_to_flags without this command having to know its keys.
    """
    string_name = string_kind.lower()
    if string_name not in ALL_STRING_FUNCTIONS:
        raise click.ClickException(
            f"String function {string_name} not found. Available: "
            f"{list(ALL_STRING_FUNCTIONS.keys())}"
        )
    ALL_STRING_FUNCTIONS[string_name].get_string(**parse_pairs(ctx.args))
