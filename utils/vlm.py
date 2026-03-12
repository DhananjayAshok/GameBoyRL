from utils.parameter_handling import load_parameters
from utils.log_handling import log_warn, log_error, log_info
from typing import List, Union
import numpy as np
import ABC
from PIL import Image
from utils.lm_inference import (
    OpenAIModel,
    OpenRouterModel,
    HuggingFaceModel,
    AnthropicModel,
)


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


def get_converted_image_list(nested_image_list) -> list[Image.Image]:
    """ "
    Converts a possibly nested list of images in either PIL Image or numpy array format into a list of PIL Images.

    :param image_list: A list of images in either PIL Image or numpy array format (H x W x C)
    :type image_list: list[Union[Image.Image, np.ndarray]]
    :return: A list of images in PIL Image format
    :rtype: list[Image.Image]
    """
    finals = []
    for item in nested_image_list:
        if isinstance(item, list):
            finals.append(get_converted_image_list(item))
        elif isinstance(item, Image.Image):
            finals.append(item)
        elif isinstance(item, np.ndarray):
            finals.append(convert_numpy_greyscale_to_pillow(item))
        else:
            log_error(
                f"Invalid image format: {type(item)}. Expected PIL Image or numpy array."
            )
    return finals


class VLM:
    def __init__(self, model_name, vlm_kind):
        """
        Initializes the VLM with the specified model and engine.

        :param model_name: The name of the model to use
        :param vlm_kind: The kind of VLM model
        """
        self._model_name = model_name
        self._vlm_kind = vlm_kind
        self._vlm = None
        if self._vlm_kind == "openai":
            self._vlm = OpenAIModel(model_name=self._model_name)
        elif self._vlm_kind == "openrouter":
            self._vlm = OpenRouterModel(model_name=self._model_name)
        elif self._vlm_kind == "huggingface":
            self._vlm = HuggingFaceModel(model_name=self._model_name, model_kind="vlm")
        elif self._vlm_kind == "anthropic":
            self._vlm = AnthropicModel(model_name=self._model_name)
        else:
            raise ValueError(f"Invalid VLM kind: {self._vlm_kind}")

    def infer(
        self,
        texts: Union[str, list[str]],
        max_new_tokens: int,
        images: Union[
            list[Union[Image.Image, np.ndarray]],
            list[list[Union[Image.Image, np.ndarray]]],
        ] = None,
    ) -> Union[str, list[str]]:
        """
        Performs inference using the VLM.

        If a single string is passed, a single string is returned. If a list is passed, a list is returned.

        :param texts: A single text prompt or a list of text prompts.
        :type texts: str or list[str]
        :param max_new_tokens: Maximum number of tokens to generate per response.
        :type max_new_tokens: int
        :param images: A list of images in either PIL Image or numpy array format (when ``texts`` is a single string) or a list of lists
            of images (when ``texts`` is a list). If None, no images are passed.
        :type images: list[Union[Image.Image, np.ndarray]] or list[list[Union[Image.Image, np.ndarray]]] or None
        :return: A single output string if ``texts`` was a string, otherwise a list of output strings.
        :rtype: str or list[str]
        """
        if images is not None:
            treated_images = get_converted_image_list(images)
            images = treated_images
        return self._vlm.infer(
            texts=texts, images=images, max_new_tokens=max_new_tokens
        )


class NamedVLM(VLM, ABC):
    NAME = None

    def __init__(self, name: str, parameters: dict = None):
        """
        Initializes the ExecutorVLM with model and kind from project parameters.
        """
        parameters = load_parameters(parameters)
        model_name = parameters[f"{name}_vlm_model"]
        vlm_kind = parameters[f"{name}_vlm_kind"]
        super().__init__(model_name=model_name, vlm_kind=vlm_kind)


class ExecutorVLM(NamedVLM):
    NAME = "executor"


class OCRVLM(NamedVLM):
    NAME = "ocr"


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
    parameters: dict = None,
) -> List[str]:
    """
    Performs OCR on the given images using the VLM.

    Args:
        images: List of images in numpy array format (H x W x C)
        vlm: The VLM instance to use. If None, uses the default ExecutorVLM.
        text_prompt: The prompt to use for the OCR model.
        do_merge: Whether to merge similar OCR results. Use this if images are sequential frames from a game.
        parameters: Optional dictionary of parameters. If None, loads project parameters.
    Returns:
        List of extracted text strings. May contain duplicates if images have frames containing the same text.
    """
    parameters = load_parameters(parameters)
    if text_prompt is None:
        text_prompt = "If there is no text in the image, just say NONE. Otherwise, perform OCR and state the text in this image:"
    max_new_tokens = parameters["ocr_max_new_tokens"]
    texts = [text_prompt] * len(images)
    if vlm is None:
        vlm = OCRVLM()
    ocred = vlm.infer(texts=texts, images=images, max_new_tokens=max_new_tokens)
    for i, res in enumerate(ocred):
        if res.strip().lower() == "none":
            log_warn(
                f"Got NONE as output from OCR. Could this have been avoided?\nimages statistics: Max: {images[i].max()}, Min: {images[i].min()}, Mean: {images[i].mean()}, percentage of non zero cells {(images[i] > 0).mean()}, percentage of non 255 cells {(images[i] < 255).mean()}",
                parameters,
            )
    ocred = [text.strip() for text in ocred if text.strip().lower() != "none"]
    if do_merge:
        ocred = merge_ocr_strings(ocred)
    return ocred
