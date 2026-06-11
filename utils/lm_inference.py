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
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import torch
import gc
import uuid

MIN_QUERIES_PER_MINUTE = 1

# Placeholder per-model rate limits (queries per minute). All currently set to
# the previous global default of 60; tune per-model as needed.
_RATE_LIMITS: dict[str, int] = {
    "gpt-4o-mini": 60,
    "gpt-4o": 60,
    "gpt-4": 60,
    "gpt-5": 60,
    "claude-opus-4.7": 60,
    "claude-sonnet-4-6": 60,
    "claude-haiku-4-5-20251001": 60,
    "google/gemini-3.1-pro-preview": 60,
    "qwen/qwen3-vl-235b-a22b-instruct": 60,
}


def get_max_queries_per_minute(model: str, parameters: dict[str, Any]) -> int:
    """
    Look up the per-model rate limit (queries per minute) from ``_RATE_LIMITS``.

    A key matches ``model`` if the key equals ``model``, the key is a substring
    of ``model``, or ``model`` is a substring of the key. If multiple keys
    match, the longest (most specific) one wins. Falls back to
    ``parameters["default_max_queries_per_minute"]`` (logging a warning) if no
    key matches.

    :param model: The model identifier string.
    :type model: str
    :param parameters: Loaded parameters dict.
    :type parameters: dict[str, Any]
    :return: The queries-per-minute limit to use for ``model``.
    :rtype: int
    """
    matches = [key for key in _RATE_LIMITS if key in model or model in key]
    if matches:
        best = max(matches, key=len)
        return _RATE_LIMITS[best]
    log_warn(
        f"Model {model} not found in _RATE_LIMITS (no exact or substring match). "
        f"Using default_max_queries_per_minute from project parameters.",
        parameters=parameters,
    )
    return parameters["default_max_queries_per_minute"]


class RateLimitedAPIBase:
    """
    Mixin that provides rate-limited API client state and ``wait()`` logic.

    Shared by ``APIModel`` and any future rate-limited API-backed model
    wrapper to avoid duplicating the init and rate-limiting code.
    """

    def __init__(
        self,
        *,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        self.parameters = load_parameters(parameters)
        self.model = model
        if max_queries_per_minute is None:
            max_queries_per_minute = get_max_queries_per_minute(model, self.parameters)
        self.max_queries_per_minute = max_queries_per_minute
        self.last_query_time = 0
        self.seconds_to_wait = 60 / self.max_queries_per_minute
        self.unique_id = str(uuid.uuid4())
        if self.max_queries_per_minute < MIN_QUERIES_PER_MINUTE:
            log_error(
                f"max_queries_per_minute must be at least {MIN_QUERIES_PER_MINUTE}, "
                f"but got {self.max_queries_per_minute}.",
                parameters=self.parameters,
            )

    def wait(self) -> None:
        """
        Enforce the rate limit by sleeping until enough time has elapsed since the last query.

        Updates ``last_query_time`` after waiting.
        """
        time_to_wait = self.seconds_to_wait - (perf_counter() - self.last_query_time)
        if time_to_wait > 0:
            sleep(time_to_wait)
        self.last_query_time = perf_counter()


class OpenAICompatibleAPIBase(RateLimitedAPIBase):
    """
    Mixin that extends ``RateLimitedAPIBase`` with an OpenAI-compatible client.

    Initializes ``self.client = OpenAI(base_url=base_url, api_key=api_key)``
    after the rate-limiting state is set up. Shared by all OpenAI-compatible
    models (``OpenAIAPIModel`` and its subclasses) to avoid repeating client
    creation in every subclass.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self.client = OpenAI(base_url=base_url, api_key=api_key)


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
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> list[list[str]]:
        """
        Run inference on a batch of text prompts with associated images. Assumes validated inputs

        :param texts: List of text prompts, one per sample.
        :type texts: list[str]
        :param images: List of image lists, one image list per sample.
        :type images: list[list[Image.Image]]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of independent sequences to return per prompt.
        :type num_return_sequences: int
        :return: Post-processed output strings shaped ``[batch, num_return_sequences]``.
        :rtype: list[list[str]]
        """
        pass

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

    def infer(
        self,
        texts: Union[str, list[str]],
        max_new_tokens: int,
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> Union[str, list[str], list[list[str]]]:
        """
        Run inference on a batch of text prompts with associated images.

        If a single string is passed, a single string is returned. If a list is passed, a list is returned.

        When ``num_return_sequences > 1``, each item is itself a list of
        ``num_return_sequences`` output strings.

        :param texts: A single text prompt or a list of text prompts.
        :type texts: str or list[str]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param images: A list of PIL Images (when ``texts`` is a single string) or a list of lists
            of PIL Images (when ``texts`` is a list). If None, no images are passed.
        :type images: list[Image.Image] or list[list[Image.Image]] or None
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of independent sequences to return per prompt.
        :type num_return_sequences: int
        :return: A single output string if ``texts`` was a string and ``num_return_sequences == 1``;
            a list of output strings if ``texts`` was a list and ``num_return_sequences == 1``;
            a list of ``num_return_sequences`` strings if ``texts`` was a string and
            ``num_return_sequences > 1``; or a list of such lists otherwise.
        :rtype: str or list[str] or list[list[str]]
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
        results = self.do_infer(texts, images, max_new_tokens, temperature=temperature, stop_strings=stop_strings, num_return_sequences=num_return_sequences)
        if num_return_sequences == 1:
            if passed_in_str:
                return results[0][0]
            else:
                return [r[0] for r in results]
        else:
            if passed_in_str:
                return results[0]
            else:
                return results

    @abstractmethod
    def infer_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
    ) -> str:
        """Run inference on a pre-formatted chat messages list. Returns a single string."""
        pass

    def partial_temperature(
        self,
        texts: Union[str, list[str]],
        max_new_tokens: int,
        switch_phrase: str,
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> Union[str, None, list]:
        """
        Run inference once at ``temperature``, then deterministically complete the
        portion of the output following ``switch_phrase``.

        Samples a "thinking" prefix at ``temperature``; if ``switch_phrase`` is found
        in the output, truncates before it and re-queries (at ``temperature=None``,
        i.e. deterministic) with the truncated output plus ``switch_phrase`` appended
        to the prompt, to obtain a deterministic final answer.

        :param texts: A single text prompt. Lists are not supported.
        :type texts: str
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param switch_phrase: Phrase marking the boundary between "thinking" and "answer".
        :type switch_phrase: str
        :param images: A list of PIL Images, or None.
        :type images: list[Image.Image] or None
        :param temperature: Sampling temperature for the first pass. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of independent sequences to return.
        :type num_return_sequences: int
        :return: The completed output string (or None if ``switch_phrase`` was not found),
            or a list of such results when ``num_return_sequences > 1``.
        :rtype: str or None or list
        """
        if isinstance(texts, list):
            log_error("partial_temperature expects a single text input, not a list.")

        first_outputs = self.infer(
            texts,
            max_new_tokens,
            images=images,
            temperature=temperature,
            stop_strings=stop_strings,
            num_return_sequences=num_return_sequences,
        )

        first_outputs_list = [first_outputs] if num_return_sequences == 1 else first_outputs

        results = []
        for output in first_outputs_list:
            if switch_phrase not in output:
                results.append(None)
                continue
            output_so_far = output.split(switch_phrase)[0]
            n_tokens_estimated = len(output_so_far.split())
            second_max_tokens = max(1, max_new_tokens - n_tokens_estimated)
            second_prompt = texts + "\nHere is what you said: " + output_so_far + "\n" + switch_phrase
            second_output = self.infer(
                second_prompt,
                second_max_tokens,
                images=images,
                temperature=None,
                stop_strings=stop_strings,
                num_return_sequences=1,
            )
            results.append(output_so_far + "\n" + switch_phrase + " " + second_output)

        if num_return_sequences == 1:
            return results[0]
        return results


class APIModel(RateLimitedAPIBase, InferenceModel, ABC):
    """
    Abstract base class for API-backed language and vision-language models.

    Handles rate limiting, image encoding, and output post-processing.
    Subclasses must implement ``get_image_input_dict``, ``query_client``,
    and ``get_output_texts``.
    """

    SUPPORTS_NATIVE_N: bool = False

    def __init__(
        self,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize the base API model with rate limiting and parameter loading.

        :param model: The model identifier string (e.g. ``"gpt-4o"``).
        :type model: str
        :param max_queries_per_minute: Maximum number of queries allowed per minute. Must be at least 1.
        :type max_queries_per_minute: Optional[int]
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
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
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            img.save(img_path, format="JPEG")
            with open(img_path, "rb") as image_file:
                encoded_images.append(
                    base64.b64encode(image_file.read()).decode("utf-8")
                )
        self.clear_encoded_image_cache(make=False)
        return encoded_images

    def clear_encoded_image_cache(self, make=True) -> str:
        """
        VLM based API models save to a cache for image processing. Clears this cache.

        :param make: Whether to create the cache directory after clearing. Default True.
        :type make: bool
        :returns: The path to the cache directory for encoded images.
        :rtype: str
        """
        cache_dir = os.path.join(self.parameters["tmp_dir"], "api_image_cache", self.unique_id)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        if make:
            os.makedirs(cache_dir)
        return cache_dir

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
    def query_client(self, messages: list[dict], max_new_tokens: int, temperature: Optional[float] = None, stop_strings: list[str] = None, num_return_sequences: int = 1) -> Any:
        """
        Send messages to the API client and return the raw response.

        :param messages: A list of message dicts formatted for the API.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of sequences to return per prompt, if natively supported.
        :type num_return_sequences: int
        :return: Response from API
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_output_texts(self, response: Any) -> list[str]:
        """
        Extract raw output text strings from a single model API response.

        :param response: The raw response object returned by the API client.
        :type response: Any
        :return: List of output text strings, one per sequence in the response.
        :rtype: list[str]
        """
        pass

    def get_outputs(self, response: Any) -> list[str]:
        """
        Extract and post-process all output texts from a single API response.

        :param response: The raw response object returned by the API client.
        :type response: Any
        :return: List of cleaned output strings, one per sequence.
        :rtype: list[str]
        """
        return [self.get_output_final(t) for t in self.get_output_texts(response)]

    def get_output(self, response: Any) -> str:
        """
        Extract and post-process the first output text from a single API response.

        :param response: The raw response object returned by the API client.
        :type response: Any
        :return: Cleaned output string.
        :rtype: str
        """
        return self.get_outputs(response)[0]

    def infer_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
    ) -> str:
        self.wait()
        response = self.query_client(messages, max_new_tokens, temperature=temperature, stop_strings=stop_strings)
        return self.get_output(response)

    def do_infer(
        self,
        texts: list[str],
        images: list[list[Image.Image]],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> list[list[str]]:
        """
        Encodes all images to base64, constructs API message dicts, enforces
        the rate limit, queries the client, and returns post-processed outputs.

        :param texts: List of text prompts, one per sample.
        :type texts: list[str]
        :param images: List of image lists, one image list per sample.
        :type images: list[list[Image.Image]]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of independent sequences to return per prompt.
        :type num_return_sequences: int
        :return: Post-processed output strings shaped ``[batch, num_return_sequences]``.
        :rtype: list[list[str]]
        """
        if len(images[0]) != 0:
            all_images = []
            for img_list in images:
                all_images.append(self.get_encoded_images(img_list))
            images = all_images
        inputs = []
        for text, img_list in zip(texts, images):
            content = [{"type": "text", "text": text}]
            for img in img_list:
                content.append(self.get_image_input_dict(img))
            inputs.append({"role": "user", "content": content})

        if num_return_sequences > 1 and temperature is None:
            log_error(
                f"num_return_sequences={num_return_sequences} requires temperature to be set "
                f"(got temperature=None); otherwise all sequences would be identical.",
                parameters=self.parameters,
            )

        self.wait()
        outputs = []
        for (
            input_message
        ) in (
            inputs
        ):  # there is no pricing advantage for batch_size > 1, so just do them sequentially to allow the caller of this function to pass lists of any size.
            if self.SUPPORTS_NATIVE_N:
                response = self.query_client([input_message], max_new_tokens, temperature=temperature, stop_strings=stop_strings, num_return_sequences=num_return_sequences)
                outputs.append(self.get_outputs(response))
            else:
                seq_outputs = []
                for seq_idx in range(num_return_sequences):
                    if seq_idx > 0:
                        self.wait()
                    response = self.query_client([input_message], max_new_tokens, temperature=temperature, stop_strings=stop_strings)
                    seq_outputs.extend(self.get_outputs(response))
                outputs.append(seq_outputs)
        return outputs


class OpenAIAPIModel(OpenAICompatibleAPIBase, APIModel):
    """
    APIModel implementation backed by an OpenAI-compatible client.

    Initializes an ``openai.OpenAI`` client pointed at the given base URL.
    Suitable as a base for any service that exposes an OpenAI-compatible API.
    """

    SUPPORTS_NATIVE_N: bool = True

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
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
        :type max_queries_per_minute: Optional[int]
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

    def query_client(self, messages: list[dict], max_new_tokens: int, temperature: Optional[float] = None, stop_strings: list[str] = None, num_return_sequences: int = 1) -> Any:
        """
        Send a message to the OpenAI chat completions endpoint.

        :param messages: A list containing the formatted user message dict.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop strings. ``"[STOP]"`` is always included.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Number of sequences to return per prompt.
        :type num_return_sequences: int
        :return: The raw API response object.
        :rtype: Any
        """
        final_stop = list(dict.fromkeys(["[STOP]"] + (stop_strings or [])))
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_new_tokens, stop=final_stop, n=num_return_sequences)
        if temperature is not None:
            kwargs["temperature"] = temperature
        max_tries = 3
        last_error = None
        for attempt in range(max_tries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                if response is None or not getattr(response, "choices", None):
                    raise ValueError(f"API returned an invalid response (None/missing/empty choices): {response}")
                return response
            except Exception as e:
                last_error = e
                log_warn(f"OpenAI API call failed on attempt {attempt+1}/{max_tries} with error: {e}")
                if attempt < max_tries - 1:
                    # exponential backoff with time_to_wait between attempts
                    backoff_time = self.seconds_to_wait * (2 ** attempt)
                    log_info(f"Waiting for {backoff_time:.2f} seconds before retrying...")
                    sleep(backoff_time)
        raise RuntimeError(f"OpenAI API call failed after {max_tries} attempts. Last error: {last_error}") from last_error

    def get_output_texts(self, response: Any) -> list[str]:
        """
        Extract output text strings from an OpenAI API response (one per choice).

        :param response: The raw response object from the OpenAI client.
        :type response: Any
        :return: List of output text strings, one per choice.
        :rtype: list[str]
        """
        texts = []
        for choice in response.choices:
            text = ""
            message = choice.message
            if hasattr(message, "reasoning") and message.reasoning is not None:
                text = "Reasoning: " + message.reasoning
            content = message.content
            if content is not None:
                if text != "":
                    text += "\nResponse: "
                text = text + " " + content
            if text.strip() == "":
                log_warn(f"Received empty output text from model for choice: {choice}")
            texts.append(text.strip())
        return texts


class OpenAIModel(OpenAIAPIModel):
    """
    Model using the official OpenAI API endpoint.

    Connects directly to OpenAI without a custom base URL.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize an OpenAI model using the default OpenAI endpoint.

        :param model: The OpenAI model identifier (e.g. ``"gpt-4o"``).
        :type model: str
        :param api_key: The OpenAI API key. If None, uses the ``OPENAI_API_KEY`` environment variable.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: Optional[int]
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


class AnthropicModel(APIModel):
    """
    APIModel implementation backed by the Anthropic Messages API.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize the Anthropic model.

        :param model: The Anthropic model identifier (e.g. ``"claude-opus-4-6"``).
        :type model: str
        :param api_key: The Anthropic API key. If None, uses the ``ANTHROPIC_API_KEY`` environment variable.
        :type api_key: str or None
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: Optional[int]
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        super().__init__(
            model=model,
            max_queries_per_minute=max_queries_per_minute,
            parameters=parameters,
        )
        self.client = Anthropic(api_key=api_key)

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

    def query_client(self, messages: list[dict], max_new_tokens: int, temperature: Optional[float] = None, stop_strings: list[str] = None, num_return_sequences: int = 1) -> Any:
        """
        Send a message to the Anthropic messages endpoint.

        :param messages: A list containing the formatted user message dict.
        :type messages: list[dict]
        :param max_new_tokens: Maximum number of tokens to generate.
        :type max_new_tokens: int
        :param temperature: Sampling temperature. None means model default.
        :type temperature: Optional[float]
        :param stop_strings: Additional stop sequences passed through to the API.
        :type stop_strings: list[str] or None
        :param num_return_sequences: Unused (Anthropic has no native multi-sample API); kept for signature compatibility.
        :type num_return_sequences: int
        :return: The raw API response object.
        :rtype: Any
        """
        kwargs = dict(model=self.model, messages=messages, max_tokens=max_new_tokens)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop_strings:
            kwargs["stop_sequences"] = stop_strings
        max_tries = 3
        last_error = None
        for attempt in range(max_tries):
            try:
                response = self.client.messages.create(**kwargs)
                if response is None or not getattr(response, "content", None):
                    raise ValueError(f"API returned an invalid response (None/missing/empty content): {response}")
                return response
            except Exception as e:
                last_error = e
                log_warn(f"Anthropic API call failed on attempt {attempt+1}/{max_tries} with error: {e}")
                if attempt < max_tries - 1:
                    # exponential backoff with time_to_wait between attempts
                    backoff_time = self.seconds_to_wait * (2 ** attempt)
                    log_info(f"Waiting for {backoff_time:.2f} seconds before retrying...")
                    sleep(backoff_time)
        raise RuntimeError(f"Anthropic API call failed after {max_tries} attempts. Last error: {last_error}") from last_error

    def get_output_texts(self, response: Any) -> list[str]:
        """
        Extract the output text string from an Anthropic API response.

        :param response: The raw response object from the Anthropic client.
        :type response: Any
        :return: A single-element list containing the output text string.
        :rtype: list[str]
        """
        text = response.content[0].text
        if text.strip() == "":
            log_warn(f"Received empty output text from model: {response}")
        return [text.strip()]


class vLLMModel(OpenAIAPIModel):
    """
    Model served via vLLM using an OpenAI-compatible API.

    Uses the OpenAI client pointed at a local or remote vLLM server. The base
    URL is read from ``parameters["vLLM_base_url"]``.
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


class OpenRouterModel(OpenAIAPIModel):
    """
    Model accessed through the OpenRouter API.

    Routes requests to various model providers (OpenAI, Anthropic, Mistral, etc.)
    via a single OpenAI-compatible endpoint at ``https://openrouter.ai/api/v1``.
    The API key is read from the ``OPENROUTER_API_KEY`` environment variable.
    """

    # OpenRouter does not reliably forward `n` to the underlying provider, so
    # multiple sequences are obtained via separate sequential calls instead.
    SUPPORTS_NATIVE_N: bool = False

    def __init__(
        self,
        model: str,
        max_queries_per_minute: Optional[int] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize an OpenRouter model.

        :param model: The OpenRouter model identifier (e.g. ``"openai/gpt-4o"``).
        :type model: str
        :param max_queries_per_minute: Maximum number of queries allowed per minute.
        :type max_queries_per_minute: Optional[int]
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
            padding=True,
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
        if self.model_kind in VLM_MODELS:
            content = [{"type": "text", "text": text}]
            for img in images:
                content.append({"type": "image", "image": img})
        else:
            content = text
        return [{"role": "user", "content": content}]

    def _generate(self, messages: list[list[dict]], max_new_tokens: int, temperature: Optional[float] = None, stop_strings: list[str] = None, num_return_sequences: int = 1):
        processor = HUGGINGFACE_MODEL_MAPPING[self.model].processor
        model = HUGGINGFACE_MODEL_MAPPING[self.model].model
        inputs = get_inputs(self.model_kind, processor, messages).to(model.device)
        tokenizer = None
        if hasattr(processor, "tokenizer"):
            tokenizer = processor.tokenizer
        else:
            tokenizer = processor
        start_index = inputs["input_ids"].shape[1]
        do_sample = temperature is not None and temperature > 0
        final_stop_strings = list(dict.fromkeys(["[STOP]"] + (stop_strings or [])))
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=None,
            temperature=temperature if do_sample else None,
            top_k=None,
            repetition_penalty=1.2,
            stop_strings=final_stop_strings,
            pad_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            num_return_sequences=num_return_sequences,
        )
        output_only = outputs[:, start_index:]
        output_texts = processor.batch_decode(output_only, skip_special_tokens=True)
        return [self.get_output_final(text.lstrip("assistant").strip()) for text in output_texts]

    def do_infer(
        self,
        texts: list[str],
        images: list[list[Image.Image]],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> list[list[str]]:
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
        flat_texts = self._generate(
            messages,
            max_new_tokens,
            temperature=temperature,
            stop_strings=stop_strings,
            num_return_sequences=num_return_sequences,
        )
        batch_size = len(texts)
        final_texts = [[] for _ in range(batch_size)]
        for i, text in enumerate(flat_texts):
            final_texts[i // num_return_sequences].append(text)
        return final_texts

    def infer_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
    ) -> str:
        if self.is_defunct:
            log_error(
                f"Cannot run inference on defunct model.",
                parameters=self.parameters,
            )
        return self._generate([messages], max_new_tokens, temperature=temperature, stop_strings=stop_strings)[0]


@dataclass
class HuggingFaceModelStore:
    model: PreTrainedModel
    processor: PreTrainedTokenizerBase | Any  # tokenizers are PreTrainedTokenizerBase; processors (e.g. AutoProcessor) have no common base
    model_kwargs: dict[str, Any]  # the kwargs the model was initialised with
    users: list["HuggingFaceModel"] = field(
        default_factory=list
    )  # all HuggingFaceModel instances currently using this model.


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
    model_kwargs.setdefault("torch_dtype", "auto")
    log_info(
        f"Loading model {model_name} of kind {model_kind} into HuggingFace model store with kwargs {model_kwargs}"
    )
    if model_kind == "lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True, **model_kwargs
        )
        processor = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
            model=model, processor=processor, model_kwargs=model_kwargs
        )
    elif model_kind == "vlm":
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True, **model_kwargs
        )
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
            model=model, processor=processor, model_kwargs=model_kwargs
        )
    elif model_kind in SPECIAL_MODEL_KINDS:
        model, processor = load_special_model(model_name, model_kind, model_kwargs)
        HUGGINGFACE_MODEL_MAPPING[model_name] = HuggingFaceModelStore(
            model=model, processor=processor, model_kwargs=model_kwargs
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


def model_factory(*, model_name: str, model_kind: str, model_engine: str, parameters: dict = None, **model_kwargs) -> InferenceModel:
    """
    Factory function to create an inference model instance based on the specified kind and engine.

    :param model_name: The identifier for the model to load (e.g. "gpt-4o", "mistral-vlm-1b").
    :type model_name: str
    :param model_kind: The kind of model to create ("lm" for language models, "vlm" for vision-language models, etc.).
    :type model_kind: str
    :param model_engine: The engine or API to use for the model ("openai", "anthropic", "huggingface", "openrouter", "vLLM", etc.).
    :type model_engine: str
    :param parameters: Loaded parameters dict. If None, loads from config.
    :type parameters: dict[str, Any] or None
    :param model_kwargs: Additional keyword arguments to pass to the model constructor.
    :type model_kwargs: dict
    :return: An instance of a subclass of InferenceModel corresponding to the specified kind and engine.
    :rtype: InferenceModel
    """
    parameters = load_parameters(parameters)
    if model_engine == "huggingface":
        return HuggingFaceModel(model=model_name, model_kind=model_kind, parameters=parameters, **model_kwargs)
    elif model_engine == "openai":
        return OpenAIModel(model=model_name, parameters=parameters, **model_kwargs)
    elif model_engine == "anthropic":
        return AnthropicModel(model=model_name, parameters=parameters, **model_kwargs)
    elif model_engine == "openrouter":
        return OpenRouterModel(model=model_name, parameters=parameters, **model_kwargs)
    elif model_engine == "vLLM":
        return vLLMModel(model=model_name, parameters=parameters, **model_kwargs)
    else:
        log_error(f"model_engine {model_engine} not recognised. Must be one of 'huggingface', 'openai', 'anthropic', 'vLLM', or 'openrouter'.", parameters=parameters)
