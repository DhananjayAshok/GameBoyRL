from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from time import sleep, perf_counter
import asyncio
import random
from typing import Optional, Any, Union
from utils.parameter_handling import load_parameters
from utils.log_handling import log_info, log_warn, log_error
import shutil
from PIL import Image
import base64
import os
from abc import ABC, abstractmethod
import uuid


def parse_key_value(text: str, key: str) -> Optional[str]:
    """
    Return the value following ``"Key:"`` on the matching line of ``text``.

    The key is matched case-insensitively; the returned value preserves its
    original case. A trailing ``[STOP]``/``[stop]`` marker is stripped.

    If ``text`` contains exactly one ``"response:"`` occurrence, only the text
    after it is searched (avoids matching mentions of ``key`` in a preceding
    "Reasoning:" section). If no ``"key:"`` line is found but ``key`` (without
    a colon) appears exactly once, the rest of that occurrence's line is
    returned instead.

    :param text: The text to search.
    :type text: str
    :param key: The key to search for.
    :type key: str
    :return: The extracted value, or None if not found or empty.
    :rtype: Optional[str]
    """
    key_lower = key.lower()
    marker = f"{key_lower}:"

    def clean(value: str) -> Optional[str]:
        value = value.strip()
        stop_idx = value.lower().find("[stop]")
        if stop_idx != -1:
            value = value[:stop_idx]
        value = value.strip()
        return value or None

    text_lower = text.lower()
    if text_lower.count("response:") == 1: # sometimes API models do this. 
        idx = text_lower.index("response:") + len("response:")
        text = text[idx:].strip()
        text_lower = text.lower()

    for line, line_lower in zip(text.splitlines(), text_lower.splitlines()):
        idx = line_lower.find(marker)
        if idx != -1:
            return clean(line[idx + len(marker):])

    if text_lower.count(key_lower) == 1:
        idx = text_lower.index(key_lower)
        rest_of_line = text[idx + len(key):].splitlines()
        return clean(rest_of_line[0]) if rest_of_line else None

    return None

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
    if not matches:
        log_warn(
            f"Model {model} not found in _RATE_LIMITS (no exact or substring match). "
            f"Using default_max_queries_per_minute from project parameters.",
            parameters=parameters,
        )
        return parameters["default_max_queries_per_minute"]
    else:
        if len(matches) > 1:
            for match in matches:
                if match == model.split("/")[-1]:
                    return _RATE_LIMITS[match]
            log_error(
                f"Multiple matches found in _RATE_LIMITS for model {model}: {matches}. "
                f"Please disambiguate by adding a more specific key to _RATE_LIMITS.",
                parameters=parameters,
            )
        else:
            return _RATE_LIMITS[matches[0]]


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
        if 0 <= self.max_queries_per_minute < MIN_QUERIES_PER_MINUTE:
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
    Mixin that extends ``RateLimitedAPIBase`` with an OpenAI-compatible async client.

    Initializes ``self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)``
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
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)


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
        output_text = output_text.split("[STOP]")[0]
        return output_text.strip()

    def _standardize_format(
        self,
        texts: Union[str, list[str]],
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
    ) -> tuple[list[str], list[list[Image.Image]], bool]:
        """
        Validates and standardizes the format of ``texts`` and ``images`` inputs as per do_infer's expectations. 

        :param texts: A single text prompt or a list of text prompts.
        :type texts: str or list[str]
        :param images: A list of PIL Images (when ``texts`` is a single string) or a list of lists
            of PIL Images (when ``texts`` is a list). If None, no images are passed.
        :type images: list[Image.Image] or list[list[Image.Image]] or None
        :return: A tuple of (standardized_texts, standardized_images, passed_in_str) where standardized_texts is a list of strings and standardized_images is a list of lists of PIL Images both formatted for input to do_infer. passed_in_str is a boolean indicating whether the original input was a single string (True) or a list of strings (False), which can be used to determine the appropriate output format in infer().
        :rtype: tuple[list[str], list[list[Image.Image]], bool]
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
        return texts, images, passed_in_str

    def infer(
        self,
        texts: Union[str, list[str]],
        max_new_tokens: int,
        images: Union[list[Image.Image], list[list[Image.Image]]] = None,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
        batch_size: int = None
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
        :param batch_size: Number of samples to process in a single batch. If None, defaults to
            ``max_batch_size_vllm``, ``max_batch_size_huggingface``, or ``max_batch_size_api`` from
            project parameters, depending on the concrete model class.
        :return: A single output string if ``texts`` was a string and ``num_return_sequences == 1``;
            a list of output strings if ``texts`` was a list and ``num_return_sequences == 1``;
            a list of ``num_return_sequences`` strings if ``texts`` was a string and
            ``num_return_sequences > 1``; or a list of such lists otherwise.
        :rtype: str or list[str] or list[list[str]]
        """
        texts, images, passed_in_str = self._standardize_format(texts, images)
        parameters = self.parameters if hasattr(self, "parameters") else load_parameters()
        if batch_size is None:
            from utils.huggingface_inference import HuggingFaceModel

            if isinstance(self, vLLMModel):
                batch_size = parameters["max_batch_size_vllm"]
            elif isinstance(self, HuggingFaceModel):
                batch_size = parameters["max_batch_size_huggingface"]
            else:
                batch_size = parameters["max_batch_size_api"]
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_images = images[i : i + batch_size]
            batch_results = self.do_infer(batch_texts, batch_images, max_new_tokens, temperature=temperature, stop_strings=stop_strings, num_return_sequences=num_return_sequences)
            results.extend(batch_results)
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
        num_return_sequences: int = 1,
    ) -> Union[str, list[str]]:
        """
        Run inference on a pre-formatted chat messages list.

        :return: A single output string if ``num_return_sequences == 1``, else a list of
            ``num_return_sequences`` output strings.
        :rtype: str or list[str]
        """
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

        :param texts: A single text prompt or a list of text prompts.
        :type texts: str or list[str]
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
        texts, images, passed_in_str = self._standardize_format(texts, images)
        asked_for_single_sequence = num_return_sequences == 1
        first_outputs = self.infer(
            texts,
            max_new_tokens,
            images=images,
            temperature=temperature,
            stop_strings=stop_strings,
            num_return_sequences=num_return_sequences,
        )

    
        first_outputs_list = [first_outputs] if asked_for_single_sequence else first_outputs # always a doubly nested list. 
        next_batch_text = []
        next_batch_images = []
        next_batch_output_so_fars = []
        next_batch_mapping = {}
        largest_max_tokens = 5
        for og_batch_i, batch_outputs in enumerate(first_outputs_list):
            for return_seq_i, output in enumerate(batch_outputs):
                if switch_phrase not in output:
                    continue
                output_so_far = output.split(switch_phrase)[0]
                n_tokens_estimated = len(output_so_far.split())
                largest_max_tokens = max(largest_max_tokens, max_new_tokens - n_tokens_estimated)
                second_prompt = texts[og_batch_i] + "\nHere is what you said: " + output_so_far + "\n" + switch_phrase + " "
                next_batch_idx = len(next_batch_text)
                next_batch_mapping[(og_batch_i, return_seq_i)] = next_batch_idx
                next_batch_text.append(second_prompt)
                next_batch_images.append(images[og_batch_i])
                next_batch_output_so_fars.append(output_so_far)

        if len(next_batch_text) == 0:
            if asked_for_single_sequence:
                if passed_in_str:
                    return None
                else:
                    return [None for _ in texts]
            else:
                if passed_in_str:
                    return [None for _ in range(num_return_sequences)]
                else:
                    return [[None for _ in range(num_return_sequences)] for _ in texts]
        second_output = self.infer(
            next_batch_text,
            largest_max_tokens,
            images=next_batch_images,
            temperature=None,
            stop_strings=stop_strings,
            num_return_sequences=1,
        )
        for i, output in enumerate(second_output):
            second_output[i] = next_batch_output_so_fars[i] + "\n" + switch_phrase + " " + output.lstrip(switch_phrase).lstrip()
        
        results = []
        for og_batch_i in range(len(first_outputs_list)):
            n_return_seqs = len(first_outputs_list[og_batch_i])
            batch_results = []
            for return_seq_i in range(n_return_seqs):
                if (og_batch_i, return_seq_i) in next_batch_mapping:
                    #breakpoint()
                    target_i = next_batch_mapping[(og_batch_i, return_seq_i)]
                    batch_results.append(second_output[target_i])
                else:
                    batch_results.append(None)
            results.append(batch_results)                
        for i, result in enumerate(results):
            if passed_in_str:
                results[i] = result[0]
        if asked_for_single_sequence:
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

        Uses a fresh per-call cache directory (rather than one shared per
        model instance) so concurrent calls from different threads on the
        same model instance don't race on each other's cached files.

        :param images: List of images in Pillow Image format.
        :type images: list[Image.Image]
        :return: List of base64 encoded image strings.
        :rtype: list[str]
        """
        cache_dir = os.path.join(
            self.parameters["tmp_dir"], "api_image_cache", self.unique_id, str(uuid.uuid4())
        )
        os.makedirs(cache_dir)
        try:
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
        finally:
            shutil.rmtree(cache_dir)
        return encoded_images

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
    async def query_client(self, messages: list[dict], max_new_tokens: int, temperature: Optional[float] = None, stop_strings: list[str] = None, num_return_sequences: int = 1) -> Any:
        """
        Send messages to the API client (asynchronously) and return the raw response.

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
        num_return_sequences: int = 1,
    ) -> Union[str, list[str]]:
        if num_return_sequences > 1 and temperature is None:
            log_error(
                f"num_return_sequences={num_return_sequences} requires temperature to be set "
                f"(got temperature=None); otherwise all sequences would be identical.",
                parameters=self.parameters,
            )
        outputs = asyncio.run(self._infer_messages_async(messages, max_new_tokens, temperature, stop_strings, num_return_sequences))
        if num_return_sequences == 1:
            return outputs[0]
        return outputs

    async def _infer_messages_async(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float],
        stop_strings: list[str],
        num_return_sequences: int,
    ) -> list[str]:
        """
        Issue the request(s) for a single chat messages list and return ``num_return_sequences``
        output strings.

        A single ``self.wait()`` paces this call relative to the last request issued;
        all ``num_return_sequences`` requests (if multiple) are then fired concurrently.
        Per-request rate-limit errors are handled by ``query_client``'s retry/backoff.
        """
        self.wait()
        if self.SUPPORTS_NATIVE_N:
            response = await self.query_client(messages, max_new_tokens, temperature=temperature, stop_strings=stop_strings, num_return_sequences=num_return_sequences)
            outputs = self.get_outputs(response)
            if len(outputs) != num_return_sequences:
                log_error(
                    f"Expected {num_return_sequences} outputs but got {len(outputs)}. Response was: {response}",
                    parameters=self.parameters,
                )
            return outputs
        else:
            async def query_one() -> str:
                response = await self.query_client(messages, max_new_tokens, temperature=temperature, stop_strings=stop_strings)
                return self.get_output(response)

            return list(await asyncio.gather(*(query_one() for _ in range(num_return_sequences))))

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

        outputs = asyncio.run(self._do_infer_async(inputs, max_new_tokens, temperature, stop_strings, num_return_sequences))
        return outputs

    async def _do_infer_async(
        self,
        inputs: list[dict],
        max_new_tokens: int,
        temperature: Optional[float],
        stop_strings: list[str],
        num_return_sequences: int,
    ) -> list[list[str]]:
        """
        Issue one query per input message concurrently and return outputs shaped
        ``[batch, num_return_sequences]``.

        A single ``self.wait()`` paces the start of this batch relative to the last
        request issued; all requests within the batch are then fired concurrently.
        Per-request rate-limit errors are handled by ``query_client``'s retry/backoff.
        """
        self.wait()
        if self.SUPPORTS_NATIVE_N:
            async def query_one(input_message: dict) -> list[str]:
                response = await self.query_client(
                    [input_message], max_new_tokens, temperature=temperature, stop_strings=stop_strings, num_return_sequences=num_return_sequences
                )
                seq_outputs = self.get_outputs(response)
                if len(seq_outputs) != num_return_sequences:
                    log_error(
                        f"Expected {num_return_sequences} outputs but got {len(seq_outputs)}. Response was: {response}",
                        parameters=self.parameters,
                    )
                return seq_outputs

            return list(await asyncio.gather(*(query_one(input_message) for input_message in inputs)))
        else:
            async def query_one(input_message: dict) -> str:
                response = await self.query_client([input_message], max_new_tokens, temperature=temperature, stop_strings=stop_strings)
                return self.get_output(response)

            flat = await asyncio.gather(*(query_one(input_message) for input_message in inputs for _ in range(num_return_sequences)))
            return [list(flat[i * num_return_sequences : (i + 1) * num_return_sequences]) for i in range(len(inputs))]


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

    async def query_client(self, messages: list[dict], max_new_tokens: int, temperature: Optional[float] = None, stop_strings: list[str] = None, num_return_sequences: int = 1) -> Any:
        """
        Send a message to the OpenAI chat completions endpoint (asynchronously).

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
                response = await self.async_client.chat.completions.create(**kwargs)
                if response is None or not getattr(response, "choices", None):
                    raise ValueError(f"API returned an invalid response (None/missing/empty choices): {response}")
                return response
            except Exception as e:
                last_error = e
                log_warn(f"OpenAI API call failed on attempt {attempt+1}/{max_tries} with error: {e}")
                if attempt < max_tries - 1:
                    # exponential backoff with time_to_wait between attempts, plus jitter so
                    # concurrent coroutines retrying after the same failure don't all collide
                    backoff_time = self.seconds_to_wait * (2 ** attempt) * random.uniform(1.0, 1.5)
                    log_info(f"Waiting for {backoff_time:.2f} seconds before retrying...")
                    await asyncio.sleep(backoff_time)
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
        self.async_client = AsyncAnthropic(api_key=api_key)

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

    async def query_client(self, messages: list[dict], max_new_tokens: int, temperature: Optional[float] = None, stop_strings: list[str] = None, num_return_sequences: int = 1) -> Any:
        """
        Send a message to the Anthropic messages endpoint (asynchronously).

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
                response = await self.async_client.messages.create(**kwargs)
                if response is None or not getattr(response, "content", None):
                    raise ValueError(f"API returned an invalid response (None/missing/empty content): {response}")
                return response
            except Exception as e:
                last_error = e
                log_warn(f"Anthropic API call failed on attempt {attempt+1}/{max_tries} with error: {e}")
                if attempt < max_tries - 1:
                    # exponential backoff with time_to_wait between attempts, plus jitter so
                    # concurrent coroutines retrying after the same failure don't all collide
                    backoff_time = self.seconds_to_wait * (2 ** attempt) * random.uniform(1.0, 1.5)
                    log_info(f"Waiting for {backoff_time:.2f} seconds before retrying...")
                    await asyncio.sleep(backoff_time)
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
        base_url: Optional[str] = None,
        parameters: dict[str, Any] = None,
    ) -> None:
        """
        Initialize a vLLM-served model.

        :param model: The model identifier as registered in the vLLM server.
        :type model: str
        :param api_key: The API key for the vLLM server, if required.
        :type api_key: str or None
        :param base_url: The base url at which the endpoint is accessible, if not default.
        :type base_url: str
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict[str, Any] or None
        """
        parameters = load_parameters(parameters)
        if base_url is None:
            base_url = parameters["vLLM_base_url"]
        super().__init__(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_queries_per_minute=-1,
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
