"""
Embedding models and in-memory vector indices.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from gameboy_worlds.emulation.parser import StateParser

from utils.log_handling import log_error, log_info
from utils.parameter_handling import load_parameters
from utils.vlm import convert_numpy_greyscale_to_pillow

_EmbeddingInput = Union[str, np.ndarray, Image.Image]


class RandomPatchProjection:
    """A learning-free image embedding: project each screen cell through a fixed random matrix.

    Game Boy specific. ``StateParser.capture_grid_cells`` splits the 160x144 screen into
    90 cells of 16x16; each is flattened, projected to ``cell_reduction_dimension``,
    normalised, and the 90 results are concatenated. The projection is seeded from
    ``random_seed`` so the same screen always embeds identically across runs.
    """

    cell_reduction_dimension = 8

    def __init__(self, parameters: dict = None):
        parameters = load_parameters(parameters)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        start = 16 * 16
        end = self.cell_reduction_dimension
        my_local_rng = torch.Generator(device=device)
        my_local_rng.manual_seed(parameters["random_seed"])
        step1 = nn.Linear(start, end, bias=False, dtype=torch.bfloat16, device=device)
        nn.init.kaiming_normal_(step1.weight, generator=my_local_rng)
        self.random_projection = nn.Sequential(step1)

    def _embed_single(self, item: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        item = np.array(item)
        grid_cells = StateParser.capture_grid_cells(item, y_offset=0)
        # Always 90 grid cells that are 16x16 (because 160 * 144)
        cell_embeddings = []
        for key in sorted(grid_cells.keys()):
            cell_image_resized = np.resize(grid_cells[key], (16, 16))
            cell_image_tensor = torch.tensor(
                cell_image_resized.flatten(),
                dtype=torch.bfloat16,
                device=self.random_projection[0].weight.device,
            )
            with torch.no_grad():
                cell_embedding = self.random_projection(cell_image_tensor)
                cell_embedding = cell_embedding / (cell_embedding.norm() + 1e-6)
            cell_embeddings.append(cell_embedding)
        # shape (cell_reduction_dimension * 90,)
        return torch.cat(cell_embeddings, dim=0)

    def project(self, items: List[Union[np.ndarray, Image.Image]]) -> torch.Tensor:
        return torch.stack([self._embed_single(item) for item in items], dim=0)


class HuggingFaceEmbeddingEngine(ABC):
    """
    Embedding engine using HuggingFace models.
    """

    MODEL_REGISTRY: Dict[str, Tuple[AutoModel, AutoProcessor, str]] = {}
    """ Model registry to cache loaded models. Keyed by model name. Value is a tuple of (model, processor, model_kind). """

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
            _model, _processor, loaded_model_kind = (
                HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name]
            )
            if loaded_model_kind != model_kind:
                log_error(
                    f"Model '{model_name}' is already loaded with model_kind "
                    f"'{loaded_model_kind}', but tried to load with different model_kind "
                    f"'{model_kind}'"
                )
            return
        log_info(f"Loading HuggingFace embedding model: {model_name} | {model_kind}")
        model, processor = engine_class._do_start(model_kind, model_name)
        HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name] = (
            model,
            processor,
            model_kind,
        )

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
        if model_kind == "qwen3":
            model = AutoModel.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="auto"
            )
            processor = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            return model, processor
        elif model_kind == "jina":
            # jina's remote code does its own device placement, so pin it to GPU directly
            # rather than through device_map.
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).to("cuda")
            return model, None
        else:
            log_error(f"Unsupported text embedding model kind: {model_kind}")

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
        if model_kind == "qwen3":
            max_length = load_parameters()["text_embedding_model_max_length"]
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
        elif model_kind == "jina":
            passage_embeddings = model.encode_text(texts=items, task="retrieval")
            return torch.stack(passage_embeddings).cpu()
        else:
            log_error(f"Text embedding model kind {model_kind} not implemented.")


class HuggingFaceImageEmbeddingEngine(HuggingFaceEmbeddingEngine):
    @staticmethod
    def _do_start(model_kind: str, model_name: str) -> Tuple[AutoModel, AutoProcessor]:
        """
        Starts the engine and returns the model and processor.
        Currently only supports Jina-Embedding models and the learning-free
        :class:`RandomPatchProjection`. Add more model kinds as needed.

        :param model_kind: The kind of model to load
        :param model_name: The name of the model to load
        :return: A tuple of (model, processor)
        :rtype: Tuple[AutoModel, AutoProcessor]
        """
        if model_kind == "jina":
            # See the note in HuggingFaceTextEmbeddingEngine._do_start.
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).to("cuda")
            return model, None
        elif model_kind == "random_patch":
            return RandomPatchProjection(), None
        else:
            log_error(f"Unsupported image embedding model kind: {model_kind}")

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
        :param items: List of input images to embed
        :type items: List[_EmbeddingInput]
        :return: Tensor containing the embeddings
        :rtype: torch.Tensor
        """
        model, _processor, _ = HuggingFaceEmbeddingEngine.MODEL_REGISTRY[model_name]
        if model_kind == "random_patch":
            return model.project(items)
        images = []
        for item in items:
            if isinstance(item, np.ndarray):
                images.append(convert_numpy_greyscale_to_pillow(item))
            elif isinstance(item, Image.Image):
                images.append(item)
            else:
                log_error(
                    f"Unsupported input type for image embedding: {type(item)}"
                )
        if model_kind == "jina":
            embeddings = model.encode_image(images=images, task="retrieval")
            return torch.stack(embeddings).cpu()
        else:
            log_error(f"Image embedding model kind {model_kind} not implemented.")


class EmbeddingModel(ABC):
    def __init__(
        self,
        model_name: str,
        model_kind: str,
        engine: Type[HuggingFaceEmbeddingEngine],
    ):
        """
        Initializes the embedding model with the specified model and engine.

        :param model_name: The name of the model to use
        :type model_name: str
        :param model_kind: The kind of Embedding model
        :type model_kind: str
        :param engine: The embedding engine class to use
        :type engine: Type[HuggingFaceEmbeddingEngine]
        """
        self._model_name = model_name
        self._model_kind = model_kind
        self._ENGINE = engine
        if not issubclass(engine, HuggingFaceEmbeddingEngine):
            log_error("EmbeddingModel only supports HuggingFaceEmbeddingEngine currently.")
        self._ENGINE.start(
            self._ENGINE, model_kind=self._model_kind, model_name=self._model_name
        )

    def embed(self, items: List[_EmbeddingInput]) -> torch.Tensor:
        """
        Generate embeddings for a list of items.

        :param items: List of inputs to embed
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
        Compare two sets of embeddings and return cosine similarity scores.

        Each row is L2-normalised before the matrix product, so entry ``(i, j)`` is the
        cosine similarity between ``embeddings_a[i]`` and ``embeddings_b[j]``.

        :param embeddings_a: First set of embeddings
        :type embeddings_a: torch.Tensor
        :param embeddings_b: Second set of embeddings
        :type embeddings_b: torch.Tensor
        :return: Similarity scores, of shape (len(embeddings_a), len(embeddings_b))
        :rtype: torch.Tensor
        """
        return torch.matmul(
            embeddings_a / embeddings_a.norm(dim=1, keepdim=True),
            (embeddings_b / embeddings_b.norm(dim=1, keepdim=True)).T,
        )

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

            - The embedding of the input. Will have the extra dimension for batch size 1.
            - The similarity scores between the input embedding and the existing index
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        embedding = self.embed([item])
        similarity_scores = self.compare(embedding, existing_index)
        return embedding, similarity_scores


class TextEmbeddingModel(EmbeddingModel):

    def __init__(self, model_name: str = None, model_kind: str = None, parameters: dict = None):
        """
        Initializes the TextEmbeddingModel with model and kind from project parameters if not provided.

        :param model_name: The name of the text embedding model to use, defaults to None
        :type model_name: str, optional
        :param model_kind: The kind of text embedding model to use, defaults to None
        :type model_kind: str, optional
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict, optional
        """
        if model_name is None:
            parameters = load_parameters(parameters)
            model_name = parameters["text_embedding_model"]
            model_kind = parameters["text_embedding_model_kind"]
        super().__init__(
            model_name=model_name,
            model_kind=model_kind,
            engine=HuggingFaceTextEmbeddingEngine,
        )


class ImageEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = None, model_kind: str = None, parameters: dict = None):
        """
        Initializes the ImageEmbeddingModel with model and kind from project parameters if not provided.

        :param model_name: The name of the image embedding model to use, defaults to None
        :type model_name: str, optional
        :param model_kind: The kind of image embedding model to use, defaults to None
        :type model_kind: str, optional
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict, optional
        """
        if model_name is None:
            parameters = load_parameters(parameters)
            model_name = parameters["image_embedding_model"]
            model_kind = parameters["image_embedding_model_kind"]
        super().__init__(
            model_name=model_name,
            model_kind=model_kind,
            engine=HuggingFaceImageEmbeddingEngine,
        )


class Index:
    def __init__(self, modality: str, parameters: dict = None):
        """
        A growable tensor of embeddings for one modality.

        :param modality: Either ``"text"`` or ``"image"``.
        :type modality: str
        :param parameters: Loaded parameters dict. If None, loads from config.
        :type parameters: dict, optional
        """
        if modality not in ["text", "image"]:
            log_error(f"Index modality must be 'text' or 'image', got {modality}")
        self.modality = modality
        if modality == "text":
            self._embedding_model = TextEmbeddingModel(parameters=parameters)
        else:
            self._embedding_model = ImageEmbeddingModel(parameters=parameters)
        self.index = None
        """ The tensor index of embeddings. Shape: (num_entries, embed_size) """

    def embed(self, items: List[_EmbeddingInput]) -> torch.Tensor:
        """
        Embed a list of items using the appropriate embedding model.

        :param items: List of input texts or images to embed
        :type items: List[_EmbeddingInput]
        :return: Tensor containing the embeddings
        :rtype: torch.Tensor
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
            self.index = torch.cat([self.index, embedding], dim=0)

    def add_to_index(self, items: List[_EmbeddingInput]):
        """
        Add new items to the index.

        :param items: List of input texts or images to add to the index
        :type items: List[_EmbeddingInput]
        """
        self.add_embedding_to_index(self.embed(items))

    def add_compare(self, item: _EmbeddingInput) -> torch.Tensor:
        """
        Embed an input and compare it to the existing index, then add it.

        :param item: The input text or image to embed and compare
        :type item: _EmbeddingInput
        :return: Similarity scores against the index as it was *before* this item was
            added, or ``None`` if the index was empty.
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


