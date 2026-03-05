from utils import (
    load_parameters,
    log_error,
    log_info,
    log_warn,
)
from execution.vlm import ExecutorVLM, convert_numpy_greyscale_to_pillow
from poke_worlds.emulation.parser import StateParser
import os
from typing import List, Tuple, Any, Dict, Union, Type
import numpy as np
from abc import ABC, abstractmethod
from PIL.Image import Image

_project_parameters = load_parameters()
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import torch
from torch import nn

# make a typing annotation named tuple that is a union of str, np.ndarray, pil.Image.Image
_EmbeddingInput = Union[str, np.ndarray, Image]


class RandomPatchProjection:
    cell_reduction_dimension = 8

    def __init__(self):
        start = 16 * 16
        end = self.cell_reduction_dimension
        my_local_rng = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        my_local_rng.manual_seed(_project_parameters["random_seed"])
        step1 = nn.Linear(
            start,
            end,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        nn.init.kaiming_normal_(step1.weight, generator=my_local_rng)
        self.random_projection = nn.Sequential(
            step1,
        )

    def _embed_single(self, item: Union[np.ndarray, Image]) -> torch.Tensor:
        item = np.array(item)
        grid_cells = StateParser.capture_grid_cells(item, y_offset=0)
        # Always 90 grid cells that are 16x16 (because 160 * 144)
        cell_embeddings = []
        cell_keys = sorted(grid_cells.keys())
        for key in cell_keys:
            cell_image = grid_cells[key]
            cell_image_resized = np.resize(cell_image, (16, 16))
            cell_image_flat = cell_image_resized.flatten()
            cell_image_tensor = torch.tensor(
                cell_image_flat,
                dtype=torch.bfloat16,
                device=self.random_projection[0].weight.device,
            )
            with torch.no_grad():
                cell_embedding = self.random_projection(cell_image_tensor)
                # normalize
                cell_embedding = cell_embedding / (cell_embedding.norm() + 1e-6)
            cell_embeddings.append(cell_embedding)
        # cell_embeddings_tensor = torch.stack(cell_embeddings, dim=0)
        # image_embedding = torch.mean(cell_embeddings_tensor, dim=0) # shape (cell_reduction_dimension,)
        image_embedding = torch.cat(
            cell_embeddings, dim=0
        )  # shape (cell_reduction_dimension*90,)
        return image_embedding

    def project(self, items: List[Union[np.ndarray, Image]]) -> torch.Tensor:
        embeddings = []
        for item in items:
            embedding = self._embed_single(item)
            embeddings.append(embedding)
        return torch.stack(embeddings, dim=0)


class HuggingFaceEmbeddingEngine(ABC):
    """
    Embedding engine using HuggingFace models.
    """

    MODEL_REGISTRY: Dict[str, Tuple[AutoModel, AutoProcessor, str]] = {}
    """ Model registry to cache loaded models. Keyed by model name. Value is a tuple of (model, processor, model_kind). """
    _DEFAULT_BATCH_SIZE = 8

    @staticmethod
    @abstractmethod
    def _do_start(model_kind: str, model_name: str) -> Tuple[AutoModel, AutoProcessor]:
        """
        Starts the engine and returns the model and processor.

        :param model_kind: The kind of model to load
        :param model_name: The name of the model to load
        :return: A tuple of (model, processor)
        :rtype: Tuple[AutoModel, AutoProcessor]
        """
        raise NotImplementedError()

    @staticmethod
    def start(
        engine_class: Type["HuggingFaceEmbeddingEngine"],
        model_kind: str,
        model_name: str,
    ):
        """
        Starts the embedding engine with the specified model.

        :param engine_class: The embedding engine class to use
        :type engine_class: Type[HuggingFaceEmbeddingEngine]
        :param model_kind: The kind of model to use
        :type model_kind: str
        :param model_name: The name of the model to use
        :type model_name: str
        """
        if model_name in HuggingFaceEmbeddingEngine.MODEL_REGISTRY:
            model, processor, _loaded_model_kind = (
                HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name]
            )
            if _loaded_model_kind != model_kind:
                log_error(
                    f"Model '{model_name}' is already loaded with model_kind '{_loaded_model_kind}', but tried to load with different model_kind '{model_kind}'",
                    _project_parameters,
                )
            else:
                return
        else:
            log_info(
                f"Loading HuggingFace VLM model: {model_name} | {model_kind}",
                _project_parameters,
            )
            model, processor = engine_class._do_start(model_kind, model_name)
            HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name] = (
                model,
                processor,
                model_kind,
            )
            return

    @staticmethod
    def is_loaded(model_name: str) -> bool:
        return model_name in HuggingFaceEmbeddingEngine.MODEL_REGISTRY

    @staticmethod
    @abstractmethod
    def _do_embed(
        model_kind: str, model_name: str, items: List[_EmbeddingInput]
    ) -> torch.Tensor:
        """
        Logic to generate embeddings for a list of items.

        :param model_kind: The kind of model to use
        :type model_kind: str
        :param model_name: The name of the model to use
        :type model_name: str
        :param items: List of input texts to embed
        :type items: List[_EmbeddingInput]
        :return: Tensor containing the embeddings
        :rtype: torch.Tensor
        """
        raise NotImplementedError()

    @staticmethod
    def embed(
        engine_class: Type["HuggingFaceEmbeddingEngine"],
        model_kind: str,
        model_name: str,
        items: List[_EmbeddingInput],
    ) -> torch.Tensor:
        """
        Generate embeddings for a list of items using the specified model.

        :param engine_class: The embedding engine class to use
        :type engine_class: Type[HuggingFaceEmbeddingEngine]
        :param model_kind: The kind of model to use
        :type model_kind: str
        :param model_name: The name of the model to use
        :type model_name: str
        :param items: List of inputs to embed
        :type items: List[_EmbeddingInput]
        :return: Tensor containing the embeddings
        :rtype: torch.Tensor
        """
        if not HuggingFaceEmbeddingEngine.is_loaded(model_name=model_name):
            HuggingFaceEmbeddingEngine.start(
                engine_class=engine_class, model_kind=model_kind, model_name=model_name
            )
        return engine_class._do_embed(
            model_kind=model_kind, model_name=model_name, items=items
        )


class HuggingFaceTextEmbeddingEngine(HuggingFaceEmbeddingEngine):

    @staticmethod
    def _do_start(model_kind: str, model_name: str) -> Tuple[AutoModel, AutoProcessor]:
        """
        Starts the engine and returns the model and processor.
        Currently only supports Qwen3-Embedding and Jina models. Add more model kinds as needed.

        :param model_kind: The kind of model to load
        :param model_name: The name of the model to load
        :return: A tuple of (model, processor)
        :rtype: Tuple[AutoModel, AutoProcessor]
        """
        # this way, we can add more model kinds w different engines (e.g. OpenAI API) later
        if model_kind not in ["qwen3", "jina"]:
            log_error(
                f"Unsupported executor_model_kind: {model_kind}", _project_parameters
            )
        if model_kind in ["qwen3"]:
            model = AutoModel.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="auto"
            )
            processor = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            return model, processor
        elif model_kind in ["jina"]:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).to("cuda")
            processor = None
            return model, processor

        else:
            log_error(
                f"Unsupported HuggingFace model kind: {model_kind}", _project_parameters
            )

    @staticmethod
    def _do_embed(
        model_kind: str, model_name: str, items: List[_EmbeddingInput]
    ) -> torch.Tensor:
        """
        Logic to generate embeddings for a list of items.

        :param model_kind: The kind of model to use
        :type model_kind: str
        :param model_name: The name of the model to use
        :type model_name: str
        :param items: List of input texts to embed
        :type items: List[_EmbeddingInput]
        :return: Tensor containing the embeddings
        :rtype: torch.Tensor
        """
        max_length = _project_parameters["text_embedding_model_max_length"]
        model, processor, _ = HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name]
        if model_kind in ["qwen3"]:
            inputs = processor(
                items,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, -1]
            return embeddings.detach().cpu()
        elif model_kind in ["jina"]:
            passage_embeddings = model.encode_text(
                texts=items,
                task="retrieval",
            )
            return torch.stack(passage_embeddings).cpu()
        else:
            log_error(
                f"Text embedding model kind {model_kind} not implemented.",
                _project_parameters,
            )


class HuggingFaceImageEmbeddingEngine(HuggingFaceEmbeddingEngine):
    @staticmethod
    def _do_start(model_kind: str, model_name: str) -> Tuple[AutoModel, AutoProcessor]:
        """
        Starts the engine and returns the model and processor.
        Currently only supports Jina-Embedding models. Add more model kinds as needed.

        :param model_kind: The kind of model to load
        :param model_name: The name of the model to load
        :return: A tuple of (model, processor)
        :rtype: Tuple[AutoModel, AutoProcessor]
        """
        # this way, we can add more model kinds w different engines (e.g. OpenAI API) later
        if model_kind not in ["jina", "random_patch"]:
            log_error(
                f"Unsupported executor_model_kind: {model_kind}", _project_parameters
            )
        if model_kind in ["jina"]:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).to("cuda")
            processor = None
            return model, processor
        elif model_kind in ["random_patch"]:
            model = RandomPatchProjection()  # dummy placeholder
            processor = None
            return model, processor
        else:
            log_error(
                f"Unsupported HuggingFace model kind: {model_kind}", _project_parameters
            )

    @staticmethod
    def _do_embed(
        model_kind: str, model_name: str, items: List[_EmbeddingInput]
    ) -> torch.Tensor:
        """
        Logic to generate embeddings for a list of items.

        :param model_kind: The kind of model to use
        :type model_kind: str
        :param model_name: The name of the model to use
        :type model_name: str
        :param items: List of input texts to embed
        :type items: List[_EmbeddingInput]
        :return: Tensor containing the embeddings
        :rtype: torch.Tensor
        """
        model, processor, _ = HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name]
        if model_kind in ["random_patch"]:
            return model.project(items)
        images = []
        for item in items:
            if isinstance(item, np.ndarray):
                pil_image = convert_numpy_greyscale_to_pillow(item)
                images.append(pil_image)
            elif isinstance(item, Image.Image):
                images.append(item)
            else:
                log_error(
                    f"Unsupported input type for image embedding: {type(item)}",
                    _project_parameters,
                )
        if model_kind in ["jina"]:
            embeddings = model.encode_image(images=images, task="retrieval")
            return torch.stack(embeddings).cpu()
        else:
            log_error(
                f"Text embedding model kind {model_kind} not implemented.",
                _project_parameters,
            )


class EmbeddingModel(ABC):
    def __init__(
        self,
        model_name: str,
        model_kind: str,
        engine: Type[HuggingFaceEmbeddingEngine],
    ):
        """
        Initializes the VLM with the specified model and engine.

        :param model_name: The name of the model to use
        :type model_name: str
        :param model_kind: The kind of Embedding model
        :type model_kind: str
        :param engine: The embedding engine class to use
        :type engine: Type[HuggingFaceEmbeddingEngine]
        """
        if not _project_parameters["vlm_importable"]:
            log_error(
                f"Tried to instantiate EmbeddingModel while vlm_importable is False. Cannot proceed. Install with the 'vlm' extra.",
                _project_parameters,
            )
        self._model_name = model_name
        self._model_kind = model_kind
        self._ENGINE = engine
        if not issubclass(engine, HuggingFaceEmbeddingEngine):
            log_error(
                f"EmbeddingModel only supports HuggingFaceEmbeddingEngine currently.",
                _project_parameters,
            )
        self._ENGINE.start(
            self._ENGINE, model_kind=self._model_kind, model_name=self._model_name
        )

    def embed(self, items: List[_EmbeddingInput]) -> torch.Tensor:
        """
        Generate text embeddings for a list of texts.

        :param items: List of input texts to embed
        :type items: List[_EmbeddingInput]
        :return: Tensor containing the embeddings
        :rtype: torch.Tensor
        """
        return self._ENGINE.embed(
            engine_class=self._ENGINE,
            model_kind=self._model_kind,
            model_name=self._model_name,
            items=items,
        )

    def compare(
        self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compare two sets of embeddings and return similarity scores.

        :param embeddings_a: First set of embeddings
        :type embeddings_a: torch.Tensor
        :param embeddings_b: Second set of embeddings
        :type embeddings_b: torch.Tensor
        :return: Similarity scores between the two sets of embeddings. Is of shape (len(embeddings_a), len(embeddings_b))
        :rtype: torch.Tensor
        """
        return torch.matmul(
            embeddings_a / embeddings_a.norm(dim=1, keepdim=True),
            (embeddings_b / embeddings_b.norm(dim=1, keepdim=True)).T,
        )  # Why does this work? TODO: Check this.

    def embed_compare(
        self, item: _EmbeddingInput, existing_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given an input and an existing index of embeddings, embed the input and compare it to the existing index.

        :param item: The input to embed and compare
        :type item: _EmbeddingInput
        :param existing_index: The existing index of embeddings to compare against
        :type existing_index: torch.Tensor
        :return: A tuple containing

            - The embedding of the input text. Will have the extra dimension for batch size 1.
            - The similarity scores between the input text embedding and the existing index
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        embedding = self.embed([item])
        similarity_scores = self.compare(embedding, existing_index)
        return embedding, similarity_scores


class TextEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name: str = None, model_kind: str = None):
        """
        Initializes the TextEmbeddingModel with model and kind from project parameters if not provided.

        :param model_name: The name of the text embedding model to use, defaults to None
        :type model_name: str, optional
        :param model_kind: The kind of text embedding model to use, defaults to None
        :type model_kind: str, optional
        """
        if model_name is None:
            model_name = _project_parameters["text_embedding_model"]
            model_kind = _project_parameters["text_embedding_model_kind"]
        super().__init__(
            model_name=model_name,
            model_kind=model_kind,
            engine=HuggingFaceTextEmbeddingEngine,
        )


class ImageEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = None, model_kind: str = None):
        """
        Initializes the ImageEmbeddingModel with model and kind from project parameters if not provided.

        :param model_name: The name of the image embedding model to use, defaults to None
        :type model_name: str, optional
        :param model_kind: The kind of image embedding model to use, defaults to None
        :type model_kind: str, optional
        """
        if model_name is None:
            model_name = _project_parameters["image_embedding_model"]
            model_kind = _project_parameters["image_embedding_model_kind"]
        super().__init__(
            model_name=model_name,
            model_kind=model_kind,
            engine=HuggingFaceImageEmbeddingEngine,
        )


def _verify_save_path(save_path: str, parameters: dict):
    if save_path is None:
        return
    storage_dir = parameters["storage_dir"]
    abs_save_path = os.path.abspath(save_path)
    abs_storage_dir = os.path.abspath(storage_dir)
    if not abs_save_path.startswith(abs_storage_dir):
        log_error(
            f"Database save path {abs_save_path} is not inside the storage directory {abs_storage_dir}.",
            parameters,
        )


class Index:
    def __init__(self, modality):
        if modality not in ["text", "image"]:
            log_error(
                f"Index modality must be 'text' or 'image', got {modality}",
                _project_parameters,
            )
        self.modality = modality
        if modality == "text":
            self._embedding_model = TextEmbeddingModel()
        else:
            self._embedding_model = ImageEmbeddingModel()
        self.index = None
        """ The tensor index of embeddings. Shape: (num_entries, embed_size) """

    def embed(self, items: List[_EmbeddingInput]) -> torch.Tensor:
        """
        Embed a list of items using the appropriate embedding model.

        :param items: List of input texts or images to embed
        :type items: List[_EmbeddingInput]
        :return: Tensor containing the embeddings
        :rtype: Tensor
        """
        if not isinstance(items, list):
            items = [items]
        return self._embedding_model.embed(items)

    def add_embedding_to_index(self, embedding: torch.Tensor):
        """
        Add a new embedding to the index.

        :param embedding: The embedding tensor to add to the index
        :type embedding: torch.Tensor
        """
        if self.index is None:
            self.index = embedding
        else:
            self.index = torch.cat([self.index, embedding], dim=0)  # TODO: Check dim

    def add_to_index(self, items: List[_EmbeddingInput]):
        """
        Add new items to the index.

        :param items: List of input texts or images to add to the index
        :type items: List[_EmbeddingInput]
        """
        new_embeddings = self.embed(items)
        self.add_embedding_to_index(new_embeddings)

    def add_compare(self, item: _EmbeddingInput) -> torch.Tensor:
        """
        Embed an input and compare it to the existing index. If the index is empty, adds the item to the index and returns None.

        :param item: The input text or image to embed and compare
        :type item: _EmbeddingInput
        :return: Similarity scores between the input embedding and the existing index
        :rtype: torch.Tensor
        """
        if self.index is None:
            self.add_to_index([item])
            return None
        embedding, similarity_scores = self._embedding_model.embed_compare(
            item, self.index
        )
        self.add_embedding_to_index(embedding)
        return similarity_scores


class DictDatabase:
    def __init__(self, save_path: str = None, parameters: dict = None):
        self._parameters = load_parameters(parameters)
        """ The mode of the database. Can be 'dense' or 'keyword'. """
        self._save_path = save_path
        """ The path to save the database to. If None, the database is not saved to disk. """
        _verify_save_path(save_path, self._parameters)
        self.data: Dict[Any, str] = {}
        """ The dictionary store of the database. Values are text values. """

    def add_entry(self, key: Any, value: str):
        """
        Add a new entry to the database.

        :param key: The key for the new entry
        :type key: Any
        :param value: The text value for the new entry
        :type value: str
        """
        if key in self.data:
            self.data[key] = self.data[key] + "| \n" + value
        else:
            self.data[key] = value
        return

    def get_entry(self, key: Any) -> str:
        """
        Get an entry from the database.

        :param key: The key for the entry to retrieve
        :type key: Any
        :return: The text value for the entry
        :rtype: str
        """
        if key not in self.data:
            return ""
        return self.data[key]

    def modify_entry(self, key: Any, new_value: str):
        """
        Modify an existing entry in the database.

        :param key: The key for the entry to modify
        :type key: Any
        :param new_value: The new text value for the entry
        :type new_value: str
        """
        if key not in self.data:
            log_error(
                f"Key {key} not found in database. Cannot modify non-existent entry.",
                self._parameters,
            )
            return
        self.data[key] = new_value
        return


class DenseTextDatabase:
    _gate_prompt = """
    You have encountered a NOVEL SITUATION of [KEY_DESCRIPTION]: '[NEW_KEY]'. 
    You may have seen a similar situation before, OLD EXPERIENCE: [OLD_KEY]
    Your task is to decide whether this old experience is relevant to the novel situation. It is relevant if [MATCH_DESCRIPTION]
    Give your output in the following format:
    Reasoning: <A very brief reasoning comparing the two situations, and reasoning whether it fits the stated criteria for relevance>
    Decision: YES or NO
    [STOP]
    Now give your answer:
    Reasoning:"""

    _gate_image = np.random.randint(
        low=0, high=255, size=(40, 40)
    )  # Placeholder image, TODO: check that this doesn't mess up the VLM.

    def __init__(
        self,
        save_path: str = None,
        parameters: dict = None,
    ):
        self._parameters = load_parameters(parameters)
        """ The mode of the database. Can be 'dense' or 'keyword'. """
        self._save_path = save_path
        """ The path to save the database to. If None, the database is not saved to disk. """
        _verify_save_path(save_path, self._parameters)
        self.keys: List[str] = []
        """ The list of text keys in the database. Is of length num_entries """
        self.values: List[Any] = []
        """ The list of values in the database. Is of length num_entries. """
        self.key_embeds: torch.Tensor = None
        """ The tensor index of embeddings for the database keys. Shape: (num_entries, embed_size) """
        self._embedding_model = TextEmbeddingModel()
        log_warn(f"Completely untested DenseTextDatabase.", self._parameters)

    def add_entries(self, entries: List[Tuple[str, Any]]):
        """
        Add new text entries to the database.

        :param entries: The list of new text entries to add. Each entry is a tuple of (key_str, value)
        :type entries: List[Tuple[str, Any]]
        """
        if len(entries) == 0:
            return
        if isinstance(entries[0], str):
            entries = [entries]  # wrap single entries
        entries_keys = [entry[0] for entry in entries]
        entries_values = [entry[1] for entry in entries]
        self.keys.extend(entries_keys)
        self.values.extend(entries_values)
        new_embeddings = self._embedding_model.embed(entries_keys)
        if self.key_embeds is None:
            self.key_embeds = new_embeddings
        else:
            self.key_embeds = torch.cat(
                [self.key_embeds, new_embeddings], dim=0
            )  # TODO: Check dim

    def get_embed_top_k(
        self, text: str, k=3
    ) -> Tuple[torch.Tensor, List[Tuple[float, str, Any]]]:
        """
        Embeds the text, and gets the top k most similar entries from the index.

        :param text: The input text to embed and compare
        :type text: str
        :param k: The number of top similar entries to retrieve, defaults to 3
        :type k: int, optional
        :return: A tuple containing

            - The embedding of the input text
            - A list of tuples for the top k entries, each tuple containing:

                - The similarity score
                - The key string
                - The value
        :rtype: Tuple[torch.Tensor, List[Tuple[float, str, Any]]]

        """
        text_embedding, similarity_scores = self._embedding_model.embed_compare(
            text, self.key_embeds
        )
        similarity_scores = similarity_scores[0]  # only one row.
        # get the argsort index
        top_k_indices = torch.topk(similarity_scores, k=k).indices.tolist()
        top_k_scores = similarity_scores[top_k_indices].tolist()
        # select the top k key strings
        top_k_keys = [self.keys[i] for i in top_k_indices]
        top_k_values = [self.values[i] for i in top_k_indices]
        top_k = list(
            zip(top_k_scores, top_k_keys, top_k_values)
        )  # List of tuples (score, key, value)
        return text_embedding, top_k

    def gate_relevance(
        self,
        new_key: str,
        top_k: List[Tuple[float, str, Any]],
        match_description: str,
        key_description: str,
        vlm_class=ExecutorVLM,
    ) -> List[Tuple[float, str, Any]]:
        """
        Given a new key and the top k similar entries, use the VLM to gate which entries are relevant.

        :param new_key: The new key string
        :type new_key: str
        :param top_k: The list of top k entries, each tuple containing (similarity score, key string, value)
        :type top_k: List[Tuple[float, str, Any]]
        :param match_description: The description of what constitutes a match (e.g. 'the old experience describes a similar visual interface as the new')
        :type match_description: str
        :param key_description: The description of the key (e.g. 'the visual interface of a game screen')
        :type key_description: str
        :param vlm_class: The VLM class to use for gating, defaults to ExecutorVLM
        :type vlm_class: class, optional
        :return: The filtered list of relevant entries from top_k
        :rtype: List[Tuple[float, str, Any]]
        """
        raise NotImplementedError("Gating not implemented yet.")
        relevant_entries = []
        for score, old_key, old_value in top_k:
            prompt = (
                self._gate_prompt.replace("[NEW_KEY]", new_key)
                .replace("[OLD_KEY]", old_key)
                .replace("[MATCH_DESCRIPTION]", match_description)
                .replace("[KEY_DESCRIPTION]", key_description)
            )
            vlm_response = vlm_class.infer(
                texts=[prompt],
                images=[self._gate_image],
                max_new_tokens=200,
            )[0].lower()
            if "decision:" in vlm_response:
                decision_declaration = vlm_response.split("decision:")[1]
                if "no" in decision_declaration:
                    continue
            relevant_entries.append((score, old_key, old_value))
        return relevant_entries

    def modify_entry(self, key: str, new_key: str, new_value: Any):
        """
        Modify an existing entry in the database. Also changes the key and recomputes embeddings.

        :param key: The key for the entry to modify
        :type key: str
        :param new_key: The new key for the entry. Repeat the old key if not changing.
        :type new_key: str
        :param new_value: The new value for the entry
        :type new_value: Any
        """
        if key not in self.keys:
            log_error(
                f"Key {key} not found in database. Cannot modify non-existent entry.",
                self._parameters,
            )
            return
        index = self.keys.index(key)
        self.values[index] = new_value
        if key != new_key:
            self.keys[index] = new_key
            new_embedding = self._embedding_model.embed([new_key])
            self.key_embeds[index : index + 1] = new_embedding
        return

    def remove_entry(self, key: str):
        """
        Remove an existing entry from the database.

        :param key: The key for the entry to remove
        :type key: str
        """
        if key not in self.keys:
            log_error(
                f"Key {key} not found in database. Cannot remove non-existent entry.",
                self._parameters,
            )
            return
        index = self.keys.index(key)
        self.keys.pop(index)
        self.values.pop(index)
        self.key_embeds = torch.cat(
            [self.key_embeds[:index], self.key_embeds[index + 1 :]], dim=0
        )  # TODO: Check dim
        return
