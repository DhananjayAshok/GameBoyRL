from utils import load_parameters, log_warn, log_error, log_info
from utils.fundamental import check_optional_installs
from typing import List, Union, Tuple, Type, Dict
import numpy as np
from PIL import Image
import torch
from abc import ABC, abstractmethod
from time import perf_counter, sleep
import os
import shutil
import base64


_project_parameters = load_parameters()
# Import anything related to vlms here.
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from openai import OpenAI
import anthropic


def convert_numpy_greyscale_to_pillow(arr: np.ndarray) -> Image:
    """
    Converts a numpy image with shape: H x W x 1 into a Pillow Image

    Args:
        arr: the numpy array

    Returns:
        image: PIL Image
    """
    rgb = np.stack([arr[:, :, 0], arr[:, :, 0], arr[:, :, 0]], axis=2)
    return Image.fromarray(rgb)


class VLMEngine(ABC):
    """
    An abstract class representing the required interface for a VLM engine.
    Implement this class to create a custom VLM engine.
    """

    @staticmethod
    @abstractmethod
    def start(**kwargs):
        """
        Starts the VLM engine and saves it to the class variables.
        After start runs successfully, is_loaded() should return True

        :param kwargs: Additional keyword arguments for specific engine implementations
        """
        pass

    @staticmethod
    @abstractmethod
    def is_loaded(**kwargs) -> bool:
        """
        Returns whether the VLM engine is loaded.

        :param kwargs: Additional keyword arguments for specific engine implementations
        :return: True if the engine is loaded, False otherwise
        :rtype: bool
        """
        pass

    @staticmethod
    @abstractmethod
    def do_infer(
        texts: List[str], images: List[np.ndarray], max_new_tokens: int, **kwargs
    ) -> List[str]:
        """
        Handles internal logic for performing inference with the given texts and images.

        :param texts: List of text prompts
        :type texts: List[str]
        :param images: List of images in numpy array format (H x W x C)
        :type images: List[np.ndarray]
        :param max_new_tokens: Maximum number of new tokens to generate
        :type max_new_tokens: int
        :param kwargs: Additional keyword arguments for specific engine implementations
        :return: List of generated text outputs
        :rtype: List[str]
        """
        pass

    @staticmethod
    @abstractmethod
    def do_multi_infer(
        texts: List[str],
        images: List[List[Union[np.ndarray, Image.Image]]],
        max_new_tokens,
        **kwargs,
    ) -> List[str]:
        """
        Handles internal logic for performing inference with the a single text and multiple images

        :param texts: List of text prompts
        :type texts: List[str]
        :param images: List of lists of images in numpy array format (H x W x C) or Pillow Image format
        :type images: List[List[Union[np.ndarray, Image.Image]]]
        :param max_new_tokens: Maximum number of new tokens to generate
        :param kwargs: Additional keyword arguments for specific engine implementations
        :return: List of generated text outputs
        :rtype: List[str]
        """
        pass

    @staticmethod
    def _prepare_infer_inputs(
        texts: Union[List[str], str], images: Union[np.ndarray, List[np.ndarray]]
    ) -> Tuple[List[str], List[np.ndarray]]:
        if isinstance(texts, str):
            texts = [texts]
            # then images must either be a single image in a list, or an array of shape (H, W, C), or a stack of shape (1, H, W, C)
            if isinstance(images, list):
                if len(images) != 1:
                    log_error(
                        f"When passing a single text string, images must be a single image in a list. Got {len(images)} images.",
                        _project_parameters,
                    )
            elif isinstance(images, np.ndarray):
                if images.ndim == 3:
                    images = [images]
                elif images.ndim == 4 and images.shape[0] == 1:
                    images = [images[0]]
                else:
                    log_error(
                        f"When passing a single text string, images must be a single image in a list or an array of shape (H, W, C) or (1, H, W, C). Got array of shape {images.shape}.",
                        _project_parameters,
                    )
            else:
                log_error(
                    f"When passing a single text string, images must be a single image in a list or an array of shape (H, W, C) or (1, H, W, C). Got type {type(images)}.",
                    _project_parameters,
                )
        return texts, images

    @staticmethod
    def _prepare_multi_infer_inputs(
        texts: Union[List[str], str],
        images: Union[
            List[List[Union[np.ndarray, Image.Image]]],
            List[Union[np.ndarray, Image.Image]],
        ],
    ) -> Tuple[List[str], List[List[Union[np.ndarray, Image.Image]]]]:
        if isinstance(texts, str):
            texts = [texts]
            # then images must be a single list of images in numpy array or Pillow Image format
            if not isinstance(images, list) or len(images) == 0:
                log_error(
                    f"When passing a single text string, images must be a nonempty list of images. Got type {type(images)}.",
                    _project_parameters,
                )
            if isinstance(images[0], list):
                log_error(
                    f"When passing a single text string, images must be a single list of images, not a list of lists. Got list of lists with length {len(images)}.",
                    _project_parameters,
                )
            images = [images]  # wrap in another list to make it a list of lists

        return texts, images

    @staticmethod
    def engine_infer(
        engine: Type["VLMEngine"],
        texts: Union[List[str], str],
        images: Union[np.ndarray, List[np.ndarray]],
        max_new_tokens: int,
        **kwargs,
    ) -> List[str]:
        """
        Performs inference with the given texts and images.

        :param engine: The VLM engine class to use for inference
        :type engine: Type[VLMEngine]
        :param texts: List of text prompts or a single text prompt
        :type texts: Union[List[str], str]
        :param images: List of images in numpy array format (H x W x C) or a single image in numpy array format
        :type images: Union[np.ndarray, List[np.ndarray]]
        :param max_new_tokens: Maximum number of new tokens to generate
        :type max_new_tokens: int
        :param kwargs: Additional keyword arguments for specific engine implementations
        :return: List of generated text outputs
        :rtype: List[str]
        """
        if not engine.is_loaded(**kwargs):
            engine.start(
                **kwargs,
            )
        texts, images = engine._prepare_infer_inputs(texts, images)
        return engine.do_infer(
            texts=texts,
            images=images,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    @staticmethod
    def multi_infer(
        engine: Type["VLMEngine"],
        texts: Union[List[str], str],
        images: Union[
            List[List[Union[np.ndarray, Image.Image]]],
            List[Union[np.ndarray, Image.Image]],
        ],
        max_new_tokens: int,
        **kwargs,
    ) -> List[str]:
        """
        Performs inference with the a single text and multiple images

        :param engine: The VLM engine class to use for inference
        :type engine: Type[VLMEngine]
        :param texts: List of text prompts or a single text prompt
        :type texts: Union[List[str], str]
        :param images: List of lists of images in numpy array format (H x W x C) or a single list of images in numpy array or Pillow Image format
        :type images: Union[List[List[Union[np.ndarray, Image.Image]]], List[Union[np.ndarray, Image.Image]]]
        :param max_new_tokens: Maximum number of new tokens to generate
        :type max_new_tokens: int
        :param kwargs: Additional keyword arguments for specific engine implementations
        :return: List of generated text outputs
        :rtype: List[str]
        """
        texts, images = VLMEngine._prepare_multi_infer_inputs(texts, images)
        if len(texts) != len(images):
            log_error(
                f"Texts and images must have the same length. Got {len(texts)} texts and {len(images)} image lists.",
                _project_parameters,
            )
        if not engine.is_loaded(**kwargs):
            engine.start(**kwargs)
        if max_new_tokens is None:
            log_error(f"Can't set max_new_tokens to None", _project_parameters)
        return engine.do_multi_infer(
            texts,
            images,
            max_new_tokens,
            **kwargs,
        )


class OpenAIVLMEngine(VLMEngine):
    """OpenAI VLM engine implementation using OpenAI API."""

    seconds_per_query = (60 / 20) + 0.01
    """ Seconds to wait between queries to avoid rate limiting. Adjust as needed."""

    _CLIENT = None
    _previous_call = 0.0

    @staticmethod
    def clear_cache() -> str:
        """Clears the OpenAI cache directory.

        Returns:
            str: The path to the temporary directory used for OpenAI cache.
        """
        tmp_dir = _project_parameters["tmp_dir"] + "/api_cache"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    @staticmethod
    def get_encoded_images(images: List[Union[np.ndarray, Image.Image]]) -> List[str]:
        """Encodes images to base64 strings for OpenAI API input.

        :param images: List of images in numpy array format (H x W x C) or Pillow Image format.
        :type images: List[Union[np.ndarray, Image.Image]]
        :return: List of base64 encoded image strings.
        :rtype: List[str]
        """
        cache_dir = OpenAIVLMEngine.clear_cache()
        encoded_images = []
        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                pil_img = convert_numpy_greyscale_to_pillow(img)
            else:
                pil_img = img
            img_path = os.path.join(cache_dir, f"image_{i}.jpg")
            pil_img.save(img_path, format="JPEG")
            with open(img_path, "rb") as image_file:
                encoded_images.append(
                    base64.b64encode(image_file.read()).decode("utf-8")
                )
        return encoded_images

    @staticmethod
    def start(**kwargs):
        OpenAIVLMEngine._CLIENT = OpenAI(api_key=_project_parameters["OPENAI_API_KEY"])

    @staticmethod
    def is_loaded(**kwargs):
        return OpenAIVLMEngine._CLIENT is not None

    @staticmethod
    def _wait(engine: Union[Type["OpenAIVLMEngine"], Type["AnthropicVLMEngine"]]):
        time_to_wait = engine.seconds_per_query - (
            perf_counter() - engine._previous_call
        )
        if time_to_wait > 0:
            sleep(time_to_wait)
        engine._previous_call = perf_counter()

    @staticmethod
    def _get_output(output) -> List[str]:
        output_texts = []
        for item in output:
            output_texts.append(item.content[0].text)
        final_texts = []
        for text in output_texts:
            if "[STOP]" in text:
                text = text.split("[STOP]")[0]
            final_texts.append(text.strip())
        return final_texts

    @staticmethod
    def do_infer(texts, images, max_new_tokens, **kwargs):
        images = OpenAIVLMEngine.get_encoded_images(images)
        base_input_dict = {
            "role": "user",
            "content": [{"type": "input_text"}, {"type": "input_image"}],
        }
        inputs = []
        for text, img in zip(texts, images):
            prompt_dict = base_input_dict.copy()
            prompt_dict["content"][0]["text"] = text
            prompt_dict["content"][1]["image_url"] = f"data:image/jpeg;base64,{img}"
            inputs.append(prompt_dict)

        model = kwargs["model_name"]
        OpenAIVLMEngine._wait(OpenAIVLMEngine)
        response = OpenAIVLMEngine._CLIENT.responses.create(
            model=model,
            input=inputs,
            max_output_tokens=max_new_tokens,
        )
        return OpenAIVLMEngine._get_output(response.output)

    @staticmethod
    def do_multi_infer(texts, images, max_new_tokens, **kwargs):
        all_images = []
        for img_list in images:
            all_images.append(OpenAIVLMEngine.get_encoded_images(img_list))
        images = all_images
        base_input_dict = {
            "role": "user",
            "content": [{"type": "input_text"}],
        }
        inputs = []
        for text, img_list in zip(texts, images):
            prompt_dict = base_input_dict.copy()
            prompt_dict["content"][0]["text"] = text
            for img in img_list:
                prompt_dict["content"].append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img}",
                    }
                )
            inputs.append(prompt_dict)

        model = kwargs["model_name"]
        OpenAIVLMEngine._wait(OpenAIVLMEngine)
        response = OpenAIVLMEngine._CLIENT.responses.create(
            model=model,
            input=inputs,
            max_output_tokens=max_new_tokens,
        )
        return OpenAIVLMEngine._get_output(response.output)


class AnthropicVLMEngine(
    VLMEngine
):  # TODO: I need to figure out how this works with multi inputs, and then make sure thats handled. rn this fails because we should be doing [response] instead
    """Anthropic VLM engine implementation using Anthropic API."""

    seconds_per_query = (60 / 20) + 0.01
    """ Seconds to wait between queries to avoid rate limiting. Adjust as needed."""

    _CLIENT = None
    _previous_call = 0.0

    @staticmethod
    def start(**kwargs):
        AnthropicVLMEngine._CLIENT = anthropic.Anthropic(
            api_key=_project_parameters["ANTHROPIC_API_KEY"]
        )

    @staticmethod
    def is_loaded(**kwargs):
        return AnthropicVLMEngine._CLIENT is not None

    @staticmethod
    def do_infer(texts, images, max_new_tokens, **kwargs):
        images = OpenAIVLMEngine.get_encoded_images(images)  # same encoding method
        base_input_dict = {
            "role": "user",
            "content": [
                {"type": "text"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg"},
                },
            ],
        }
        inputs = []
        for text, img in zip(texts, images):
            prompt_dict = base_input_dict.copy()
            prompt_dict["content"][0]["text"] = text
            prompt_dict["content"][1]["source"]["data"] = img
            inputs.append(prompt_dict)

        model = kwargs["model_name"]
        OpenAIVLMEngine._wait(AnthropicVLMEngine)
        response = AnthropicVLMEngine._CLIENT.messages.create(
            model=model,
            messages=inputs,
            max_tokens=max_new_tokens,
        )
        return OpenAIVLMEngine._get_output(response)

    @staticmethod
    def do_multi_infer(texts, images, max_new_tokens, **kwargs):
        all_images = []
        for img_list in images:
            all_images.append(OpenAIVLMEngine.get_encoded_images(img_list))
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
                prompt_dict["content"].append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img,
                        },
                    }
                )
            inputs.append(prompt_dict)

        model = kwargs["model_name"]
        OpenAIVLMEngine._wait(AnthropicVLMEngine)
        response = AnthropicVLMEngine._CLIENT.messages.create(
            model=model,
            messages=inputs,
            max_tokens=max_new_tokens,
        )
        return OpenAIVLMEngine._get_output(response)


class HuggingFaceVLMEngine(VLMEngine):
    """
    A VLM engine implementation using HuggingFace transformers.
    Currently only supports Qwen3-VL models.
    """

    MODEL_REGISTRY: Dict[
        str, Tuple[AutoModelForImageTextToText, AutoProcessor, str]
    ] = {}
    """ Model registry to cache loaded models. Keyed by model name. Value is a tuple of (model, processor, vlm_kind). """
    _DEFAULT_BATCH_SIZE = 8

    @staticmethod
    def _do_start(
        vlm_kind: str, model_name: str
    ) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
        """
        Starts the VLM engine and returns the model and processor.
        Currently only supports Qwen3-VL models. Add more model kinds as needed.

        :param vlm_kind: The kind of VLM model to load
        :param model_name: The name of the model to load
        :return: A tuple of (model, processor)
        :rtype: Tuple[AutoModelForImageTextToText, AutoProcessor]
        """
        # this way, we can add more model kinds w different engines (e.g. OpenAI API) later
        if vlm_kind not in ["qwen3vl"]:
            log_error(f"Unsupported executor_vlm_kind: {vlm_kind}", _project_parameters)
        if vlm_kind in ["qwen3vl"]:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
            return model, processor
        else:
            log_error(
                f"Unsupported HuggingFace VLM kind: {vlm_kind}", _project_parameters
            )

    @staticmethod
    def _get_kwargs(**kwargs):
        required_keys = ["model_name", "vlm_kind"]
        for key in required_keys:
            if key not in kwargs:
                log_error(
                    f"Missing required argument '{key}' for HuggingFaceVLMEngine.start()",
                    _project_parameters,
                )
        model_name = kwargs["model_name"]
        vlm_kind = kwargs["vlm_kind"]
        return model_name, vlm_kind

    @staticmethod
    def start(**kwargs):
        model_name, vlm_kind = HuggingFaceVLMEngine._get_kwargs(**kwargs)
        if model_name in HuggingFaceVLMEngine.MODEL_REGISTRY:
            model, processor, _loaded_vlm_kind = HuggingFaceVLMEngine.MODEL_REGISTRY[
                model_name
            ]
            if _loaded_vlm_kind != vlm_kind:
                log_error(
                    f"Model '{model_name}' is already loaded with vlm_kind '{_loaded_vlm_kind}', but tried to load with different vlm_kind '{vlm_kind}'",
                    _project_parameters,
                )
            else:
                return
        else:
            log_info(
                f"Loading HuggingFace VLM model: {model_name}", _project_parameters
            )
            model, processor = HuggingFaceVLMEngine._do_start(vlm_kind, model_name)
            HuggingFaceVLMEngine.MODEL_REGISTRY[model_name] = (
                model,
                processor,
                vlm_kind,
            )
            return

    @staticmethod
    def do_infer(texts, images, max_new_tokens, **kwargs):
        model_name, _ = HuggingFaceVLMEngine._get_kwargs(**kwargs)
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        else:
            batch_size = HuggingFaceVLMEngine._DEFAULT_BATCH_SIZE
        model, processor, vlm_kind = HuggingFaceVLMEngine.MODEL_REGISTRY[model_name]
        if vlm_kind == "qwen3vl":
            all_images = [convert_numpy_greyscale_to_pillow(img) for img in images]
            all_texts = [
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{text}\n<|im_end|><|im_start|>assistant\n"
                for text in texts
            ]
            all_outputs = []
            for i in range(0, len(all_images), batch_size):
                images = all_images[i : i + batch_size]
                texts = all_texts[i : i + batch_size]
                inputs = processor(
                    text=texts,
                    images=images,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(model.device)
                input_length = inputs["input_ids"].shape[1]
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_p=None,
                    temperature=None,
                    top_k=None,
                    repetition_penalty=1.2,
                    stop_strings=["[STOP]"],
                    tokenizer=processor.tokenizer,
                )
                output_only = outputs[:, input_length:]
                decoded_outputs = processor.batch_decode(
                    output_only, skip_special_tokens=True
                )
                all_outputs.extend(decoded_outputs)
            return all_outputs
        else:
            log_error(
                f"Unsupported HuggingFace VLM kind: {vlm_kind}",
                _project_parameters,
            )

    @staticmethod
    def do_multi_infer(texts, images, max_new_tokens, **kwargs):
        model_name, _ = HuggingFaceVLMEngine._get_kwargs(**kwargs)
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        else:
            batch_size = HuggingFaceVLMEngine._DEFAULT_BATCH_SIZE
        model, processor, vlm_kind = HuggingFaceVLMEngine.MODEL_REGISTRY[model_name]
        if vlm_kind == "qwen3vl":
            all_outputs = []
            all_texts = []
            for i, text in enumerate(texts):
                full_text = f"<|im_start|>user\n"
                for j in range(len(images[i])):
                    full_text += (
                        f"Picture: {i+1}<|vision_start|><|image_pad|><|vision_end|>"
                    )
                full_text += f"\n{text}\n<|im_end|><|im_start|>assistant\n"
                all_texts.append(full_text)
            for i in range(0, len(all_texts), batch_size):
                batch_images = images[i : i + batch_size]
                flat_images = []
                for img_list in batch_images:
                    for img in img_list:
                        if isinstance(img, np.ndarray):
                            flat_images.append(convert_numpy_greyscale_to_pillow(img))
                        else:
                            flat_images.append(img)
                batch_texts = all_texts[i : i + batch_size]
                inputs = processor(
                    text=batch_texts,
                    images=flat_images,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(model.device)
                input_length = inputs["input_ids"].shape[1]
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_p=None,
                    temperature=None,
                    top_k=None,
                    repetition_penalty=1.2,
                    stop_strings=["[STOP]"],
                    tokenizer=processor.tokenizer,
                )
                output_only = outputs[:, input_length:]
                decoded_outputs = processor.batch_decode(
                    output_only, skip_special_tokens=True
                )
                all_outputs.extend(decoded_outputs)
            return all_outputs
        else:
            log_error(
                f"Unsupported HuggingFace VLM kind: {vlm_kind}",
                _project_parameters,
            )

    @staticmethod
    def is_loaded(**kwargs):
        model_name, _ = HuggingFaceVLMEngine._get_kwargs(**kwargs)
        return model_name in HuggingFaceVLMEngine.MODEL_REGISTRY


class VLM:
    def __init__(self, model_name, vlm_kind, engine=None):
        """
        Initializes the VLM with the specified model and engine.

        :param model_name: The name of the model to use
        :param vlm_kind: The kind of VLM model
        :param engine: The VLM engine class to use (subclass of VLMEngine). If None, defaults based on vlm_kind.
        """
        self._model_name = model_name
        self._vlm_kind = vlm_kind
        self._standard_kwargs = {
            "model_name": self._model_name,
            "vlm_kind": self._vlm_kind,
        }
        self._ENGINE = None
        if engine is not None and not issubclass(engine, VLMEngine):
            log_error(
                f"engine must be a subclass of VLMEngine. Got {engine}",
                _project_parameters,
            )
        if engine is not None:
            self._ENGINE = engine
        else:
            if self._vlm_kind == "openai":
                self._ENGINE = OpenAIVLMEngine
            elif self._vlm_kind == "anthropic":
                self._ENGINE = AnthropicVLMEngine
            else:
                self._ENGINE = HuggingFaceVLMEngine
        self._ENGINE.start(**self._standard_kwargs)

    def infer(
        self,
        texts: Union[List[str], str],
        images: Union[np.ndarray, List[np.ndarray]],
        max_new_tokens: int,
    ) -> List[str]:
        """
        Performs inference with the given texts and images

        :param texts: List of text prompts or a single text prompt
        :type texts: Union[List[str], str]
        :param images: List of images in numpy array format (H x W x C) or a single image in numpy array format
        :type images: Union[np.ndarray, List[np.ndarray]]
        :param max_new_tokens: Maximum number of new tokens to generate
        :type max_new_tokens: int
        :return: List of generated text outputs
        :rtype: List[str]
        """
        return self._ENGINE.engine_infer(
            self._ENGINE,
            texts,
            images,
            max_new_tokens,
            **self._standard_kwargs,
        )

    def multi_infer(
        self,
        texts: Union[List[str], str],
        images: Union[
            List[List[Union[np.ndarray, Image.Image]]],
            List[Union[np.ndarray, Image.Image]],
        ],
        max_new_tokens: int,
    ) -> List[str]:
        """
        Performs inference with the a single text and multiple images

        :param texts: List of text prompts or a single text prompt
        :type texts: Union[List[str], str]
        :param images: List of lists of images in numpy array format (H x W x C) or a single list of images in numpy array or Pillow Image format
        :type images: Union[List[List[Union[np.ndarray, Image.Image]]], List[Union[np.ndarray, Image.Image]]]
        :param max_new_tokens: Maximum number of new tokens to generate
        :type max_new_tokens: int
        :return: List of generated text outputs
        :rtype: List[str]
        """
        return self._ENGINE.multi_infer(
            self._ENGINE,
            texts,
            images,
            max_new_tokens,
            **self._standard_kwargs,
        )


class ExecutorVLM(VLM):
    """
    A class that holds the VLM for the executor
    """

    def __init__(self):
        """
        Initializes the ExecutorVLM with model and kind from project parameters.
        """
        self._model_name = _project_parameters["executor_vlm_model"]
        self._vlm_kind = _project_parameters["executor_vlm_kind"]
        super().__init__(model_name=self._model_name, vlm_kind=self._vlm_kind)


def merge_ocr_strings(strings, min_overlap=3):
    """
    Merges a list of strings by removing subsets and combining overlapping fragments.

    Written by Gemini3 Pro, but it seems to work.

    Args:
        strings (list): List of strings from OCR.
        min_overlap (int): Minimum characters required to consider two strings an overlap.
    """
    # 1. Clean up: Remove exact duplicates and empty strings
    current_strings = list(set(s.strip() for s in strings if s.strip()))

    # 2. Remove subsets (if "Hello" is in "Hello World", remove "Hello")
    # Sorting by length descending ensures we check smaller strings against larger ones
    current_strings.sort(key=len, reverse=True)
    final_set = []
    for s in current_strings:
        if not any(s in other for other in final_set):
            final_set.append(s)

    # 3. Iterative Overlap Merging
    # We use a while loop because merging two strings might create a new
    # string that can then be merged with a third string.
    merged_list = final_set[:]
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(merged_list):
            j = 0
            while j < len(merged_list):
                if i == j:
                    j += 1
                    continue

                s1, s2 = merged_list[i], merged_list[j]

                # Check if suffix of s1 matches prefix of s2
                overlap_len = 0
                max_possible_overlap = min(len(s1), len(s2))

                for length in range(max_possible_overlap, min_overlap - 1, -1):
                    if s1.endswith(s2[:length]):
                        overlap_len = length
                        break

                if overlap_len > 0:
                    # Create the merged string
                    new_string = s1 + s2[overlap_len:]

                    # Remove the two old strings and add the new one
                    # We use indices carefully or rebuild the list
                    val_i = merged_list[i]
                    val_j = merged_list[j]
                    merged_list.remove(val_i)
                    merged_list.remove(val_j)
                    merged_list.append(new_string)

                    changed = True
                    # Reset indices to restart search with the new combined string
                    i = -1
                    break
                j += 1
            if changed:
                break
            i += 1

    return merged_list


def ocr(
    images: List[np.ndarray],
    *,
    vlm: VLM = None,
    text_prompt=None,
    do_merge: bool = True,
) -> List[str]:
    """
    Performs OCR on the given images using the VLM.

    Args:
        images: List of images in numpy array format (H x W x C)
        vlm: The VLM instance to use. If None, uses the default ExecutorVLM.
        text_prompt: The prompt to use for the OCR model.
        do_merge: Whether to merge similar OCR results. Use this if images are sequential frames from a game.
    Returns:
        List of extracted text strings. May contain duplicates if images have frames containing the same text.
    """
    if text_prompt is None:
        text_prompt = "If there is no text in the image, just say NONE. Otherwise, perform OCR and state the text in this image:"
    parameters = _project_parameters
    max_new_tokens = parameters["ocr_max_new_tokens"]
    texts = [text_prompt] * len(images)
    if vlm is None:
        vlm = ExecutorVLM()
    ocred = vlm.infer(texts=texts, images=images, max_new_tokens=max_new_tokens)
    for i, res in enumerate(ocred):
        if res.strip().lower() == "none":
            log_warn(
                f"Got NONE as output from OCR. Could this have been avoided?\nimages statistics: Max: {images[i].max()}, Min: {images[i].min()}, Mean: {images[i].mean()}, percentage of non zero cells {(images[i] > 0).mean()}, percentage of non 255 cells {(images[i] < 255).mean()}",
                _project_parameters,
            )
    ocred = [text.strip() for text in ocred if text.strip().lower() != "none"]
    if do_merge:
        ocred = merge_ocr_strings(ocred)
    return ocred
