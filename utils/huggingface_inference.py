from utils.lm_inference import *
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from transformers import (
    AutoModel,
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
        model_kind: str = None,
        parameters: dict[str, Any] = None,
        **model_kwargs,
    ) -> None:
        if model_kind is None:
            model_kind = infer_model_kind(model, error_out=True)
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
    ) -> list[str]:
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
        final_texts = []
        for text in output_texts:
            # HF includes the stop string in the output; strip at the earliest hit.
            earliest = len(text)
            for stop in final_stop:
                idx = text.find(stop)
                if idx != -1 and idx < earliest:
                    earliest = idx
            if earliest < len(text):
                text = text[:earliest]
            final_texts.append(text.lstrip("assistant").strip())
        return final_texts

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
        num_return_sequences: int = 1,
    ) -> Union[str, list[str]]:
        if self.is_defunct:
            log_error(
                f"Cannot run inference on defunct model.",
                parameters=self.parameters,
            )
        outputs = self._generate(
            [messages],
            max_new_tokens,
            temperature=temperature,
            stop_strings=stop_strings,
            num_return_sequences=num_return_sequences,
        )
        if num_return_sequences == 1:
            return outputs[0]
        return outputs


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
