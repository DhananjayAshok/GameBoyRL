"""
Sometimes in bash, there are times when you want to call functions and get string values back.
This script is a helper to get those string values from bash functions. It takes in the name of the string and the arguments to pass to that function, and it prints the output of the function to stdout.
If there is an error at any point of time, it will print the error message to stderr and exit with a non-zero exit code.

This script should ONLY print once, as the output of the function. If there are multiple print statements, it will be difficult to parse the output and get the desired string value.
"""

import sys
from abc import ABC, abstractmethod


def log(message):
    print(message, file=sys.stderr)


def depathify(string) -> str:
    """
    Helper function to convert a path-like string to a string that can be used as a filename or an experiment name.

    :param string: the string to depathify
    :type string: str
    :return: the depathified string
    :rtype: str
    """
    return (
        string.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


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
        "buffer_load_path",
        "seed",
    ]

    def add_to_exp_name(self, exp_name, keys, kwargs):
        for i, key in enumerate(keys):
            exp_name += f"{key}_{kwargs[key]}"
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
            exp_name += f"-{depathify(kwargs['embedder_load_path'])}_"
        else:
            exp_name += f"_"
        exp_name = self.add_to_exp_name(
            exp_name, keys=["similarity_metric", "curiosity_module"], kwargs=kwargs
        )
        if kwargs["buffer_load_path"] is not None:
            exp_name += f"-{depathify(kwargs['buffer_load_path'])}_"
        else:
            exp_name += f"_"
        exp_name += f"seed_{kwargs['seed']}"
        return exp_name


class EnvID(StringFunction):
    NAME = "env_id"
    REQUIRED_ARGS = ["game", "env", "init_state", "controller", "max_steps"]

    def _get_string(self, **kwargs) -> str:
        return f"gameboy_worlds-{kwargs['game']}-{kwargs['env']}-{kwargs['init_state']}-{kwargs['controller']}-{kwargs['max_steps']}"


STRING_FUNCTIONS = [ExperimentName, EnvID]

ALL_STRING_FUNCTIONS = {func.NAME.lower(): func() for func in STRING_FUNCTIONS}


def parse():
    passed_in_args = sys.argv[1:]
    if len(passed_in_args) < 1:
        raise ValueError("Must provide at least the string name to get")
    string_name = passed_in_args[0].lower()
    if string_name not in ALL_STRING_FUNCTIONS:
        raise ValueError(
            f"String function {string_name} not found. Available string functions: {list(ALL_STRING_FUNCTIONS.keys())}"
        )
    args = passed_in_args[1:]
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
    return string_name, arg_dict


if __name__ == "__main__":
    string_name, args = parse()
    ALL_STRING_FUNCTIONS[string_name].get_string(**args)
