from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

from utils.lm_inference import InferenceModel, _collapse_meta
from utils.log_handling import log_error, log_info, log_warn
from utils.parameter_handling import load_parameters
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


class HuggingFaceModelBase(ABC):
    """
    Abstract base for all HuggingFace model wrappers (inference and embedding).

    Manages registration in the shared ``HUGGINGFACE_MODEL_MAPPING`` store,
    tracks model kwargs, and marks instances as defunct when the underlying
    model is evicted from GPU memory.
    """

    def _init_store(self, *, model: str, parameters, model_kwargs: dict, load_fn) -> None:
        """Register this instance in the shared store, loading the model if needed."""
        self.model = model
        self.parameters = load_parameters(parameters)
        self.model_kwargs = model_kwargs
        self.is_defunct = False
        if model in HUGGINGFACE_MODEL_MAPPING:
            existing_kwargs = HUGGINGFACE_MODEL_MAPPING[model].model_kwargs
            if existing_kwargs != model_kwargs:
                log_warn(
                    f"Model {model} already loaded with different kwargs. "
                    f"Passed: {model_kwargs}, loaded: {existing_kwargs}. Reloading. "
                    "Existing instances of this model will be marked defunct."
                )
                remove_from_model_store(model)
                load_fn(model_name=model, model_kwargs=model_kwargs)
        else:
            load_fn(model_name=model, model_kwargs=model_kwargs)
        HUGGINGFACE_MODEL_MAPPING[model].users.append(self)


class HuggingFaceModel(HuggingFaceModelBase, InferenceModel):
    def __init__(
        self,
        model: str,
        model_kind: str,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        if model_kind not in SPECIAL_MODEL_KINDS:
            log_error(
                f"model_kind {model_kind!r} not recognised. Must be one of: {SPECIAL_MODEL_KINDS}",
                parameters=parameters,
            )
        self.model_kind = model_kind
        self._init_store(
            model=model,
            parameters=parameters,
            model_kwargs=model_kwargs,
            load_fn=lambda *, model_name, model_kwargs: load_model_into_store(model_name, model_kind, model_kwargs),
        )

    def get_single_message_list(self, text: str, images: list[Image.Image]) -> dict:
        if self.model_kind in VLM_MODELS:
            content = [{"type": "text", "text": text}]
            for img in images:
                content.append({"type": "image", "image": img})
        else:
            content = text
        return [{"role": "user", "content": content}]

    def _generate(
        self,
        messages: list[list[dict]],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> tuple[list[str], list[tuple[Optional[int], int]]]:
        """
        Generate for a batch of message lists and report per-sequence token counts.

        Both returned lists are flat with length ``batch * num_return_sequences``, ordered
        ``[r0s0, r0s1, ..., r1s0, ...]`` (row ``j`` belongs to record
        ``j // num_return_sequences``).

        Token counts are **pre-trim**: they describe what the model actually generated,
        before the ``[STOP]``/stop-string truncation applied to the text below. Input
        tokens are the record's real prompt length (padding excluded) and are reported on
        the record's first row only, with ``0`` on its remaining rows, since the prompt is
        encoded once per record however many sequences are sampled from it. Output tokens
        count generated tokens up to the terminating EOS, excluding it (generation pads
        with ``eos_token_id``, so the two are indistinguishable here).

        :return: ``(final_texts, usage_stats)`` where ``usage_stats[j]`` is
            ``(input_tokens, output_tokens)`` for ``final_texts[j]``.
        :rtype: tuple[list[str], list[tuple[Optional[int], int]]]
        """
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
        if num_return_sequences > 1 and not do_sample:
            log_error(
                "num_return_sequences > 1 requires temperature sampling (temperature must be > 0).",
                parameters=self.parameters,
            )
        final_stop = list(dict.fromkeys(["[STOP]"] + (stop_strings or [])))
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=None,
            temperature=temperature if do_sample else None,
            top_k=None,
            repetition_penalty=1.2,
            stop_strings=final_stop,
            pad_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,
            num_return_sequences=num_return_sequences,
        )
        output_only = outputs[:, start_index:]
        output_texts = processor.batch_decode(output_only, skip_special_tokens=True)

        if "attention_mask" in inputs:
            record_input_tokens = [int(count) for count in inputs["attention_mask"].sum(dim=1)]
        else:
            # No mask means nothing was padded, so every record used the full width.
            record_input_tokens = [start_index] * inputs["input_ids"].shape[0]
        pad_token_id = tokenizer.eos_token_id  # what generate() was told to pad with
        if isinstance(pad_token_id, (list, tuple)):
            pad_token_id = pad_token_id[0]
        usage_stats = []
        for row_index, row in enumerate(output_only.tolist()):
            n_generated = len(row)
            for token_index, token in enumerate(row):
                if token == pad_token_id:
                    n_generated = token_index
                    break
            record_index = row_index // num_return_sequences
            is_first_row_of_record = row_index % num_return_sequences == 0
            usage_stats.append(
                (record_input_tokens[record_index] if is_first_row_of_record else 0, n_generated)
            )

        final_texts = []
        for text in output_texts:
            # HF includes the stop string in the output; strip at the earliest hit.
            lowered = text.lower()
            earliest = len(text)
            for stop in final_stop:
                idx = lowered.find(stop.lower())
                if idx != -1 and idx < earliest:
                    earliest = idx
            if earliest < len(text):
                text = text[:earliest]
            final_texts.append(text.lstrip("assistant").strip())
        return final_texts, usage_stats

    def do_infer(
        self,
        texts: list[str],
        images: list[list[Image.Image]],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> dict[str, Any]:
        """
        Run local HuggingFace generation on a batch. Parameters are as
        :meth:`InferenceModel.do_infer`.

        Token counts come from the generation tensors (see :meth:`_generate`) and carry
        two HuggingFace-specific caveats worth knowing when comparing them to an API
        backend's: they are **pre-trim** (what the model actually generated, before the
        ``[STOP]``/stop-string truncation applied to the returned text), and output tokens
        **exclude** the terminating EOS. The prompt is encoded once per record however
        many sequences are sampled, so ``input_tokens`` does not scale with
        ``num_return_sequences``.

        :return: ``{"output": ..., "meta": ...}`` where ``output`` holds the post-processed
            output strings shaped ``[batch, num_return_sequences]`` and ``meta`` is
            ``{"input_tokens": [...], "output_tokens": [...]}``, each a list of length
            ``batch`` (one entry per record, output tokens summed over the record's
            sequences). See :meth:`InferenceModel._build_meta`.
        :rtype: dict[str, Any]
        """
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
        flat_texts, flat_usages = self._generate(
            messages,
            max_new_tokens,
            temperature=temperature,
            stop_strings=stop_strings,
            num_return_sequences=num_return_sequences,
        )
        batch_size = len(texts)
        final_texts = [[] for _ in range(batch_size)]
        final_usages = [[] for _ in range(batch_size)]
        for i, text in enumerate(flat_texts):
            final_texts[i // num_return_sequences].append(text)
            final_usages[i // num_return_sequences].append(flat_usages[i])
        return {"output": final_texts, "meta": self._build_meta(usages=final_usages)}

    def infer_messages(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        stop_strings: list[str] = None,
        num_return_sequences: int = 1,
    ) -> dict[str, Any]:
        """
        Run local HuggingFace generation on a pre-formatted chat messages list.

        The same pre-trim / EOS-excluded token-counting caveats as :meth:`do_infer` apply.

        :return: ``{"output": ..., "meta": ...}``. ``output`` is a single output string if
            ``num_return_sequences == 1``, else a list of ``num_return_sequences`` output
            strings. A messages list is a single record, so ``meta``'s
            ``input_tokens``/``output_tokens`` are scalars (int or None) regardless of
            ``num_return_sequences``, with output tokens summed over the sequences.
        :rtype: dict[str, Any]
        """
        if self.is_defunct:
            log_error(
                f"Cannot run inference on defunct model.",
                parameters=self.parameters,
            )
        outputs, usages = self._generate(
            [messages],
            max_new_tokens,
            temperature=temperature,
            stop_strings=stop_strings,
            num_return_sequences=num_return_sequences,
        )
        meta = _collapse_meta(self._build_meta(usages=[usages]))
        if num_return_sequences == 1:
            return {"output": outputs[0], "meta": meta}
        return {"output": outputs, "meta": meta}


@dataclass
class HuggingFaceModelStore:
    model: PreTrainedModel
    processor: PreTrainedTokenizerBase | Any  # tokenizers are PreTrainedTokenizerBase; processors (e.g. AutoProcessor) have no common base
    model_kwargs: dict[str, Any]
    users: list["HuggingFaceModelBase"] = field(default_factory=list)


# Single shared store for all HuggingFace model wrappers (inference and embedding).
HUGGINGFACE_MODEL_MAPPING: dict[str, HuggingFaceModelStore] = {}


def set_users_defunct(model_name: str) -> None:
    if model_name in HUGGINGFACE_MODEL_MAPPING:
        for user in HUGGINGFACE_MODEL_MAPPING[model_name].users:
            user.is_defunct = True
    else:
        log_warn(f"Model {model_name} not found in store. Cannot set users defunct.")


def remove_from_model_store(model_name: str, verbose: bool = False) -> None:
    """Remove a model from the store and free its GPU memory."""
    if model_name in HUGGINGFACE_MODEL_MAPPING:
        if verbose:
            log_info(f"Removing {model_name} from store and clearing from GPU.")
        set_users_defunct(model_name)
        store = HUGGINGFACE_MODEL_MAPPING.pop(model_name)
        for param in store.model.parameters():
            param.data = torch.empty(0, device=param.device)
        del store
        gc.collect()
        torch.cuda.empty_cache()
    elif verbose:
        log_warn(f"Model {model_name} not found in store. Cannot remove.")


def clear_model_store() -> None:
    """Remove all models from the store and free their GPU memory."""
    for model_name in list(HUGGINGFACE_MODEL_MAPPING.keys()):
        remove_from_model_store(model_name)


def load_special_model(model, model_kind, model_kwargs):
    raise NotImplementedError


def load_model_into_store(model_name, model_kind, model_kwargs) -> None:
    remove_from_model_store(model_name, verbose=False)
    log_info(
        f"Loading model {model_name} of kind {model_kind} into HuggingFace model store with kwargs {model_kwargs}"
    )
    if model_kind == "lm":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True, **model_kwargs
        )
        processor = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
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
