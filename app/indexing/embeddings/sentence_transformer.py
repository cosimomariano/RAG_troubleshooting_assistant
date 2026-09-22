from collections.abc import Sequence
from typing import Protocol

from app.indexing.embeddings.base import EmbeddingVector


class SentenceTransformerBackend(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Sequence[Sequence[float]]: ...


class SentenceTransformerEmbeddingModel:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        backend: SentenceTransformerBackend | None = None,
    ) -> None:
        self._model_name = self._validate_model_name(model_name)
        self._batch_size = self._validate_batch_size(batch_size)
        self._normalize_embeddings = normalize_embeddings
        self._backend = self._resolve_backend(backend)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def normalize_embeddings(self) -> bool:
        return self._normalize_embeddings

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        text_batch = list(texts)
        if not text_batch:
            return []

        encoded_vectors = self._backend.encode(
            text_batch,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
        )
        vectors = self._convert_vectors(encoded_vectors)
        self._validate_result_count(text_batch, vectors)
        return vectors

    def _resolve_backend(
        self,
        backend: SentenceTransformerBackend | None,
    ) -> SentenceTransformerBackend:
        if backend is not None:
            return backend
        return self._load_backend(self._model_name)

    @staticmethod
    def _convert_vectors(
        encoded_vectors: Sequence[Sequence[float]],
    ) -> list[EmbeddingVector]:
        vectors: list[EmbeddingVector] = []
        for encoded_vector in encoded_vectors:
            vector = [float(value) for value in encoded_vector]
            vectors.append(vector)
        return vectors

    @staticmethod
    def _validate_result_count(
        text_batch: list[str],
        vectors: list[EmbeddingVector],
    ) -> None:
        if len(vectors) != len(text_batch):
            raise ValueError(
                "Il numero di embedding restituiti non coincide con il numero di testi."
            )

    @staticmethod
    def _validate_model_name(model_name: str) -> str:
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("Il nome del modello di embedding non può essere vuoto.")
        return normalized_model_name

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("La dimensione del batch deve essere maggiore di zero.")
        return batch_size

    @staticmethod
    def _load_backend(model_name: str) -> SentenceTransformerBackend:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Impossibile caricare Sentence Transformers.") from error

        return SentenceTransformer(model_name)
