# These are all the utils functions or classes that you may want to import in your project.
#
# Resolved LAZILY (PEP 562 module __getattr__), not imported eagerly. `from utils import X`
# still works exactly as before; the difference is that nothing is loaded until something is
# actually touched.
#
# This matters because eager imports here made *every* import of *any* utils submodule pay for
# the whole VLM stack. Python executes a package's __init__ before any of its submodules, so
# `from utils.parameter_handling import load_parameters` was loading torch, transformers,
# openai, anthropic, PIL and numpy — measured at ~15s on this cluster's filesystem, against
# 0.08s for a bare interpreter. python_funcs.py shells out once per path lookup and cannot
# afford that; nor can `--help`.
#
# To add a symbol: put it in _EXPORTS under the submodule that defines it.

import importlib

_EXPORTS = {
    "utils.parameter_handling": ("load_parameters", "compute_secondary_parameters"),
    "utils.log_handling": ("log_error", "log_info", "log_warn"),
    "utils.fundamental": ("file_makedir", "depathify"),
    "utils.vlm": (
        "ExecutorVLM",
        "convert_numpy_greyscale_to_pillow",
        "VLM",
        "ocr",
        "object_detection",
        "identify_matches",
    ),
    "utils.huggingface_inference": ("HuggingFaceModel",),
    "utils.lm_inference": (
        "InferenceModel",
        "OpenAIModel",
        "OpenRouterModel",
        "AnthropicModel",
        "vLLMModel",
        "parse_key_value",
        "parse_yes_no",
        "sum_meta",
        "sum_optional",
        "zero_meta",
    ),
    "utils.parsing": (
        "PLAN_SEPARATOR",
        "parse_action_line",
        "parse_int",
        "parse_list",
        "parse_plan",
        "parse_steps",
    ),
}

#: symbol -> module that defines it
_SYMBOL_SOURCE = {
    name: module for module, names in _EXPORTS.items() for name in names
}

__all__ = sorted(_SYMBOL_SOURCE)


def __getattr__(name: str):
    """Import the defining submodule on first access, then cache the symbol on the package."""
    module_name = _SYMBOL_SOURCE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'utils' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # subsequent lookups skip __getattr__ entirely
    return value


def __dir__():
    return __all__
