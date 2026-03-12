from openai import OpenAI
from anthropic import Anthropic
from time import sleep, perf_counter
from typing import Optional, Any, Union
from utils.parameter_handling import load_parameters
from utils.log_handling import log_info, log_warn, log_error
import shutil
from PIL import Image
import base64
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
)
import torch
import gc

MIN_QUERIES_PER_MINUTE = 1


class InferenceModel(ABC):
    """
    Abstract base class for all LM inference that support inference
    """

    @abstractmethod
    def do_infer(
        self,
        texts: list[str],
        max_new_tokens: int,
        images: list[list[Image.Image]] = None,
    ) -> list[str]:
        """
        Run inference on a batch of text prompts with associated images. Assumes validated inputs

        :param texts: List of text prompts, one per sample.
        :type texts: list[str]
        :param images: List of image lists, one image list per sample.
        :type images: list[list[Image.Image]]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :return: Post-processed output strings, one per sample.
        :rtype: list[str]
        """
        pass

    def infer(
        self,
        texts: Union[str, list[str]],
        max_new_tokens: int,
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
    ) -> Union[str, list[str]]:
        """
        Run inference on a batch of text prompts with associated images.

        If a single string is passed, a single string is returned. If a list is passed, a list is returned.

        :param texts: A single text prompt or a list of text prompts.
        :type texts: str or list[str]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param images: A list of PIL Images (when ``texts`` is a single string) or a list of lists
            of PIL Images (when ``texts`` is a list). If None, no images are passed.
        :type images: list[Image.Image] or list[list[Image.Image]] or None
        :return: A single output string if ``texts`` was a string, otherwise a list of output strings.
        :rtype: str or list[str]
        """
        passed_in_str = isinstance(texts, str)
        if passed_in_str:
            if texts.strip() == "":
                log_error(f"texts cannot be empty")
            texts = [texts]
        else:
            if not isinstance(texts, list):
                log_error(
                    f"texts must be a string or list of strings. Got {type(texts)}"
                )
            if len(texts) == 0:
                log_error(f"texts cannot be empty.")
            for item in texts:
                if not isinstance(item, str):
                    log_error(f"Got {type(item)}:{item} instead of str as text")

        if images is not None:
            if not isinstance(images, list):
                log_error(
                    f"images must be a list of PIL images or list of lists of PIL Images. Got {type(images)}"
                )
            else:
                if len(images) == 0:
                    log_error(f"images cannot be empty")
                if passed_in_str:
                    for item in images:
                        if not isinstance(item, Image.Image):
                            if isinstance(item, list):
                                log_error(
                                    f"Passed in a single string for texts but  list of lists for images. This is confusing."
                                )
                            log_error(
                                f"image list contains non images: {type(item)}: {item}"
                            )
                    images = [images]
                else:
                    for list_item in images:
                        if not isinstance(list_item, list):
                            log_error(
                                f"images must be a list of list of PIL Images, got a {type(list_item)}:{list_item}"
                            )
                        for item in list_item:
                            if not isinstance(item, Image.Image):
                                log_error(
                                    f"image list contains non images: {type(item)}: {item}"
                                )
            if len(texts) != len(images):
                log_error(
                    f"Number of text prompts and number of image lists must be the same. Got {len(texts)} text prompts and {len(images)} image lists."
                )
        else:
            images = [[] for _ in texts]
        results = self.do_infer(texts, images, max_new_tokens)
        if passed_in_str:
            return results[0]
        else:
            return results


class APIModel(InferenceModel, ABC):
    """
    Abstract base class for API-backed language and vision-language models.

    Handles rate limiting, image encoding, and output post-processing.
    Subclasses must implement ``get_image_input_dict``, ``query_client``,
    and ``get_output_texts``.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize the base API model with rate limiting and parameter loading.

        :param model: The model identifier string (e.g. ``"gpt-4o"``).
        :type model: str
        :param base_url: The base URL for the API endpoint.
        :type base_url: str
        :param api_key: The API key for authentication. If None, uses environment variables.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute. Must be at least 1.
        :type max_queries_per_minute: int
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        self.parameters = load_parameters(parameters)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_queries_per_minute = max_queries_per_minute
        self.last_query_time = 0
        self.seconds_to_wait = 60 / self.max_queries_per_minute
        # error out if max_queries_per_minute is less than limit
        if self.max_queries_per_minute < MIN_QUERIES_PER_MINUTE:
            log_error(
                f"max_queries_per_minute must be at least {MIN_QUERIES_PER_MINUTE}, but got {self.max_queries_per_minute}.",
                parameters=self.parameters,
            )

    def get_encoded_images(self, images: list[Image.Image]) -> list[str]:
        """Encodes images to base64 strings for OpenAI API input.

        :param images: List of images in Pillow Image format.
        :type images: list[Image.Image]
        :return: List of base64 encoded image strings.
        :rtype: list[str]
        """
        cache_dir = self.clear_encoded_image_cache()
        encoded_images = []
        for i, img in enumerate(images):
            img_path = os.path.join(cache_dir, f"image_{i}.jpg")
            img.save(img_path, format="JPEG")
            with open(img_path, "rb") as image_file:
                encoded_images.append(
                    base64.b64encode(image_file.read()).decode("utf-8")
                )
        return encoded_images

    def clear_encoded_image_cache(self) -> str:
        """
        VLM based API models save to a cache for image processing. Clears this cache.

        :returns: The path to the cache directory for encoded images.
        :rtype: str
        """
        cache_dir = os.path.join(self.parameters["tmp_dir"], "api_image_cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        return cache_dir

    def wait(self) -> None:
        """
        Enforce the rate limit by sleeping until enough time has elapsed since the last query.

        Updates ``last_query_time`` after waiting.
        """
        time_to_wait = self.seconds_to_wait - (perf_counter() - self.last_query_time)
        if time_to_wait > 0:
            sleep(time_to_wait)
        self.last_query_time = perf_counter()
        return

    def get_output_final(self, output_text: str) -> str:
        """
        Post-process a single output text by truncating at the ``[STOP]`` token and stripping whitespace.

        :param output_text: Raw output string from the model.
        :type output_text: str
        :return: Cleaned output string with content after ``[STOP]`` removed.
        :rtype: str
        """
        if "[STOP]" in output_text:
            output_text = output_text.split("[STOP]")[0]
        return output_text.strip()

    @abstractmethod
    def get_image_input_dict(self, image: str) -> dict:
        """
        Return the API-specific content dict for a single base64-encoded image.

        :param image: A base64-encoded image string.
        :type image: str
        :return: A dictionary formatted for inclusion in the API message content.
        :rtype: dict
        """
        pass

    @abstractmethod
    def query_client(self, messages: list[dict], max_new_tokens: int) -> Any:
        """
        Send messages to the API client and return raw response texts.

        :param messages: A list of message dicts formatted for the API.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :return: Response from API
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_output_texts(self, response: Any) -> str:
        """
        Extract raw output text strings from a single model API response.

        :param response: The raw response object returned by the API client.
        :type response: Any
        :return: output text strings.
        :rtype: str
        """
        pass

    def get_output(self, response: Any) -> str:
        """
        Extract and post-process the output text from a single API response.

        Calls ``get_output_texts`` to extract the raw text, then applies
        ``get_output_final`` to clean it.

        :param response: The raw response object returned by the API client.
        :type response: Any
        :return: Cleaned output string.
        :rtype: str
        """
        output_text = self.get_output_texts(response)
        return self.get_output_final(output_text)

    def do_infer(
        self,
        texts: list[str],
        images: list[list[Image.Image]],
        max_new_tokens: int,
    ) -> list[str]:
        """
        Encodes all images to base64, constructs API message dicts, enforces
        the rate limit, queries the client, and returns post-processed outputs.

        :param texts: List of text prompts, one per sample.
        :type texts: list[str]
        :param images: List of image lists, one image list per sample.
        :type images: list[list[Image.Image]]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :return: Post-processed output strings, one per sample.
        :rtype: list[str]
        """
        if len(images[0]) != 0:
            all_images = []
            for img_list in images:
                all_images.append(self.get_encoded_images(img_list))
            images = all_images
        base_input_dict = {
            "role": "user",
            "content": [{"type": "text"}],
        }
        inputs = []
        for text, img_list in zip(texts, images):
            prompt_dict = base_input_dict.copy()
            prompt_dict["content"][0]["text"] = text
            for img in img_list:
                prompt_dict["content"].append(self.get_image_input_dict(img))
            inputs.append(prompt_dict)

        self.wait()
        responses = []
        for (
            input_message
        ) in (
            inputs
        ):  # there is no pricing advantage for batch_size > 1, so just do them sequentially to allow the caller of this function to pass lists of any size.
            response = self.query_client([input_message], max_new_tokens)
            responses.append(response)
        outputs = []
        for response in responses:
            outputs.append(self.get_output(response))
        return outputs


class OpenAIAPIModel(APIModel):
    """
    APIModel implementation backed by an OpenAI-compatible client.

    Initializes an ``openai.OpenAI`` client pointed at the given base URL.
    Suitable as a base for any service that exposes an OpenAI-compatible API.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize the OpenAI-compatible API model.

        :param model: The model identifier string.
        :type model: str
        :param base_url: The base URL for the OpenAI-compatible API endpoint.
        :type base_url: str
        :param api_key: The API key for authentication. If None, uses environment variables.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: int
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def get_image_input_dict(self, image: str) -> dict:
        """
        Return the OpenAI-format content dict for a base64-encoded image.

        :param image: A base64-encoded JPEG image string.
        :type image: str
        :return: A content dict with ``type`` and ``image_url`` fields.
        :rtype: dict
        """
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        }

    def query_client(self, messages: list[dict], max_new_tokens: int) -> Any:
        """
        Send a message to the OpenAI chat completions endpoint.

        :param messages: A list containing the formatted user message dict.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :return: The raw API response object.
        :rtype: Any
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
        )
        return response

    def get_output_texts(self, response: Any) -> str:
        """
        Extract the output text string from an OpenAI API response.

        :param response: The raw response object from the OpenAI client.
        :type response: Any
        :return: The output text string.
        :rtype: str
        """
        text = ""
        message = response.choices[0].message
        if hasattr(message, "reasoning"):
            text = "Reasoning: " + message.reasoning
        content = response.choices[0].message.content
        if content is not None:
            if text != "":
                text += "\nResponse: "
            text = text + " " + content
        if text.strip() == "":
            log_warn(f"Received empty output text from model: {response}")
        return text.strip()


class OpenAIModel(OpenAIAPIModel):
    """
    Model using the official OpenAI API endpoint.

    Connects directly to OpenAI without a custom base URL.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize an OpenAI model using the default OpenAI endpoint.

        :param model: The OpenAI model identifier (e.g. ``"gpt-4o"``).
        :type model: str
        :param api_key: The OpenAI API key. If None, uses the ``OPENAI_API_KEY`` environment variable.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: int
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            base_url=None,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)


class AnthropicModel(APIModel):
    """
    Anthropic API model (stub — not yet implemented).

    The client is currently uninitialized. Implement ``get_image_input_dict``,
    ``query_client``, and ``get_output_texts`` using the Anthropic SDK before use.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize the Anthropic model stub.

        :param model: The Anthropic model identifier (e.g. ``"claude-opus-4-6"``).
        :type model: str
        :param api_key: The Anthropic API key. If None, uses the ``ANTHROPIC_API_KEY`` environment variable.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: int
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            base_url=None,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self.client = Anthropic(api_key=self.api_key)

    def get_image_input_dict(self, image: str) -> dict:
        """
        Return the Anthropic-format content dict for a base64-encoded image.

        :param image: A base64-encoded JPEG image string.
        :type image: str
        :return: A content dict with ``type`` and ``source`` fields.
        :rtype: dict
        """
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image,
            },
        }

    def query_client(self, messages: list[dict], max_new_tokens: int) -> Any:
        """
        Send a message to the Anthropic messages endpoint.

        :param messages: A list containing the formatted user message dict.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :return: The raw API response object.
        :rtype: Any
        """
        response = self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
        )
        return response

    def get_output_texts(self, response: Any) -> str:
        return response.content[0].text


class vLLMModel(OpenAIAPIModel):
    """
    Model served via vLLM using an OpenAI-compatible API.

    Uses the OpenAI client pointed at a local or remote vLLM server.

    .. note::
        The default base URL is not yet configured. Set ``base_url`` before use.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize a vLLM-served model.

        :param model: The model identifier as registered in the vLLM server.
        :type model: str
        :param api_key: The API key for the vLLM server, if required.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: int
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        parameters = load_parameters(parameters)
        base_url = parameters["vLLM_base_url"]
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)


class OpenRouterModel(OpenAIAPIModel):
    """
    Model accessed through the OpenRouter API.

    Routes requests to various model providers (OpenAI, Anthropic, Mistral, etc.)
    via a single OpenAI-compatible endpoint at ``https://openrouter.ai/api/v1``.
    """

    def __init__(
        self,
        model: str,
        max_queries_per_minute: int = 60,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize an OpenRouter model.

        :param model: The OpenRouter model identifier (e.g. ``"openai/gpt-4o"``).
        :type model: str
        :param api_key: The OpenRouter API key.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: int
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        api_key = os.environ["OPENROUTER_API_KEY"]
        super().__init__(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)


SPECIAL_MODEL_KINDS = ["lm", "vlm"]
VLM_MODELS = ["vlm"]

for model in VLM_MODELS:
    if model not in SPECIAL_MODEL_KINDS:
        log_error(
            f"Model {model} is in VLM_MODELS but not in SPECIAL_MODEL_KINDS. Please add it to SPECIAL_MODEL_KINDS."
        )


def get_inputs(model_kind, processor, messages):
    if model_kind in ["lm", "vlm"]:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs
    else:
        raise NotImplementedError


class HuggingFaceModel(InferenceModel):
    def __init__(
        self,
        model: str,
        model_kind: str = None,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        self.model = model
        if model_kind is None:
            model_kind = infer_model_kind(model, error_out=True)
        self.model_kind = model_kind
        self.model_kwargs = model_kwargs
        self.parameters = parameters
        self.is_defunct = False
        if model in HUGGINGFACE_MODEL_MAPPING:
            used_kwargs = HUGGINGFACE_MODEL_MAPPING[model].model_kwargs
            if used_kwargs != model_kwargs:
                log_warn(
                    f"Model {model} already loaded with different kwargs. Passed kwargs: {model_kwargs}, loaded kwargs: {used_kwargs}. Reloading model with new kwargs. This will make existing HuggingFaceModel instances using this model defunct."
                )
                remove_from_huggingface_model_store(model)
                load_model_into_store(model, model_kind, model_kwargs)
        else:
            load_model_into_store(model, model_kind, model_kwargs)
        HUGGINGFACE_MODEL_MAPPING[model].users.append(self)

    def get_single_message_list(self, text: str, images: list[Image.Image]) -> dict:
        content = [{"type": "text", "text": text}]
        if images is not None:
            for img in images:
                content.append(
                    {
                        "type": "image",
                        "image": img,
                    }
                )
        return [{"role": "user", "content": content}]

    def do_infer(
        self,
        texts: list[str],
        images: list[list[Image.Image]],
        max_new_tokens: int,
    ) -> list[str]:
        if self.is_defunct:
            log_error(
                f"Cannot run inference on defunct model.",
                parameters=self.parameters,
            )
        if len(images[0]) != 0 and self.model_kind not in VLM_MODELS:
            log_error(
                f"Model {self.model} of kind {self.model_kind} cannot handle image inputs."
            )
        messages = [
            self.get_single_message_list(text, img_list)
            for text, img_list in zip(texts, images)
        ]
        processor = HUGGINGFACE_MODEL_MAPPING[self.model].processor
        model = HUGGINGFACE_MODEL_MAPPING[self.model].model
        inputs = get_inputs(self.model_kind, processor, messages).to(model.device)
        tokenizer = None
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor
        start_index = inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            temperature=None,
            top_k=None,
            repetition_penalty=1.2,
            stop_strings=["[STOP]"],
            pad_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
        )
        output_only = outputs[:, start_index:]
        output_texts = processor.batch_decode(output_only, skip_special_tokens=True)
        final_texts = []
        for text in output_texts:
            final_texts.append(text.lstrip("assistant").strip())
        return final_texts


@dataclass
class HuggingFaceModelStore:
    model: AutoModel
    processor: AutoProcessor
    model_kwargs: dict[str, Any]  # the kwargs the model was initialised with
    users: list[
        HuggingFaceModel
    ]  # all HuggingFaceModel instances currently using this model.


# Stores all HuggingFace models loaded into GPU. Allows different instantiations
# of HuggingFaceModel to reuse a loaded model without reloading it every time.
HUGGINGFACE_MODEL_MAPPING: dict[str, HuggingFaceModelStore] = {}


def set_users_defunct(model: str) -> None:
    """
    Mark all HuggingFaceModel instances using the given model name as defunct.

    This should be called before removing a model from the store to prevent
    future inference calls on HuggingFaceModel instances that rely on the removed model.

    :param model: The name of the model whose users should be marked defunct.
    :type model: str
    """
    if model in HUGGINGFACE_MODEL_MAPPING:
        for user in HUGGINGFACE_MODEL_MAPPING[model].users:
            user.is_defunct = True
    else:
        log_warn(
            f"Model {model} not found in HuggingFace model store with keys: {HUGGINGFACE_MODEL_MAPPING.keys()}. Cannot set users defunct."
        )


def remove_from_huggingface_model_store(model: str, verbose=False) -> None:
    """
    Remove a model from the HuggingFace model store and free its GPU memory.

    Zeros out all parameter tensors in-place on the GPU (avoiding a CPU transfer),
    then forces Python GC and flushes PyTorch's CUDA cache.
    Does nothing if ``model`` is not in the store.

    :param model: The name of the model to remove.
    :type model: str
    :param verbose: If True, logs removal and missing-key warnings.
    :type verbose: bool
    """
    if model in HUGGINGFACE_MODEL_MAPPING:
        if verbose:
            log_info(
                f"Removing {model} from HuggingFace model store and clearing from GPU."
            )
        set_users_defunct(model)
        store = HUGGINGFACE_MODEL_MAPPING.pop(model)
        for param in store.model.parameters():
            param.data = torch.empty(0, device=param.device)
        del store
        gc.collect()
        torch.cuda.empty_cache()
    else:
        if verbose:
            log_warn(
                f"Model {model} not found in HuggingFace model store with keys: {HUGGINGFACE_MODEL_MAPPING.keys()}. Cannot remove."
            )


def clear_huggingface_model_store() -> None:
    """
    Remove all models from the HuggingFace model store and free their GPU memory.
    """
    for model in list(HUGGINGFACE_MODEL_MAPPING.keys()):
        remove_from_huggingface_model_store(model)


def load_special_model(model, model_kind, model_kwargs):
    raise NotImplementedError


def load_model_into_store(model_name, model_kind, model_kwargs) -> None:
    remove_from_huggingface_model_store(model_name, verbose=False)
    log_info(
        f"Loading model {model_name} of kind {model_kind} into HuggingFace model store with kwargs {model_kwargs}"
    )
    if model_kind == "lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", **model_kwargs
        )
        processor = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
            model=model, processor=processor, model_kwargs=model_kwargs, users=[]
        )
    elif model_kind == "vlm":
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map="auto", **model_kwargs
        )
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
        HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
            model=model, processor=processor, model_kwargs=model_kwargs, users=[]
        )
    elif model_kind in SPECIAL_MODEL_KINDS:
        model, processor = load_special_model(model_name, model_kind, model_kwargs)
        HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
            model=model, processor=processor, model_kwargs=model_kwargs, users=[]
        )
    else:
        log_error(
            f"Model class {model_kind} not recognised. Cannot load model {model_name} into store."
        )


def infer_model_kind(model: str, error_out: bool = False) -> Optional[str]:
    special_inclusions = []
    for special_class in SPECIAL_MODEL_KINDS:
        if special_class.lower() in model.lower():
            special_inclusions.append(special_class)
    if len(special_inclusions) == 1:
        return special_inclusions[0]
    if error_out:
        log_error(
            f"Could not infer model class for {model}. Specify `model_kind` as one of: {SPECIAL_MODEL_KINDS}"
        )
    return None
