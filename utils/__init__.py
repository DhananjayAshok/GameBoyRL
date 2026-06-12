# These are all the utils functions or classes that you may want to import in your project
from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils.log_handling import log_error, log_info, log_warn, log_dict
from utils.hash_handling import write_meta, add_meta_details
from utils.plot_handling import Plotter
from utils.fundamental import file_makedir
from utils.tests import paired_bootstrap
from utils.vlm import ExecutorVLM, convert_numpy_greyscale_to_pillow, VLM, ocr, object_detection, identify_matches
from utils.huggingface_inference import HuggingFaceModel
from utils.lm_inference import (
    InferenceModel,
    OpenAIModel,
    OpenRouterModel,
    AnthropicModel,
    vLLMModel,
    parse_key_value,
)


def model_factory(*, model_name: str, model_kind: str, parameters: dict = None, **model_kwargs) -> InferenceModel:
    """
    Instantiate an InferenceModel by provider name.

    :param model_name: Model identifier (e.g. "gpt-4o-mini", "openai/gpt-4o-mini").
    :param model_kind: Provider string — one of "openai", "openrouter", "anthropic", "vllm", "huggingface".
    :param parameters: Loaded parameters dict. If None, loads from config.
    :param model_kwargs: Extra kwargs forwarded to the model constructor.
    """
    parameters = load_parameters(parameters)
    if model_kind == "huggingface":
        from utils.huggingface_inference import HuggingFaceModel
        return HuggingFaceModel(model=model_name, parameters=parameters, **model_kwargs)
    elif model_kind == "openai":
        return OpenAIModel(model=model_name, parameters=parameters, **model_kwargs)
    elif model_kind == "openrouter":
        return OpenRouterModel(model=model_name, parameters=parameters, **model_kwargs)
    elif model_kind == "anthropic":
        return AnthropicModel(model=model_name, parameters=parameters, **model_kwargs)
    elif model_kind == "vllm":
        return vLLMModel(model=model_name, parameters=parameters, **model_kwargs)
    else:
        log_error(
            f"model_kind {model_kind!r} not recognised. "
            "Must be one of 'openai', 'openrouter', 'anthropic', 'vllm', 'huggingface'.",
            parameters=parameters,
        )
