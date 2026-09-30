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
        modelName: str,
        *,
        batchSize: int = 32,
        normalizeEmbeddings: bool = True,
        inputPrefix: str = "",
        backend: SentenceTransformerBackend | None = None,
    ) -> None:
        self.modelName = self.validateModelName(modelName)
        self.batchSize = self.validateBatchSize(batchSize)
        self.normalizeEmbeddings = normalizeEmbeddings
        self.inputPrefix = inputPrefix
        self.backend = self.resolveBackend(backend)

    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        textBatch = [self.inputPrefix + text for text in texts]
        if not textBatch:
            return []

        encodedVectors = self.backend.encode(
            textBatch,
            batch_size=self.batchSize,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalizeEmbeddings,
        )
        vectors = self.convertVectors(encodedVectors)
        self.validateResultCount(textBatch, vectors)
        return vectors

    def resolveBackend(
        self,
        backend: SentenceTransformerBackend | None,
    ) -> SentenceTransformerBackend:
        if backend is not None:
            return backend
        return self.loadBackend(self.modelName)

    @staticmethod
    def convertVectors(
        encodedVectors: Sequence[Sequence[float]],
    ) -> list[EmbeddingVector]:
        vectors: list[EmbeddingVector] = []
        for encodedVector in encodedVectors:
            vector = [float(value) for value in encodedVector]
            vectors.append(vector)
        return vectors

    @staticmethod
    def validateResultCount(
        textBatch: list[str],
        vectors: list[EmbeddingVector],
    ) -> None:
        if len(vectors) != len(textBatch):
            raise ValueError(
                "Il numero di embedding restituiti non coincide con il numero di testi."
            )

    @staticmethod
    def validateModelName(modelName: str) -> str:
        normalizedModelName = modelName.strip()
        if not normalizedModelName:
            raise ValueError("Il nome del modello di embedding non può essere vuoto.")
        return normalizedModelName

    @staticmethod
    def validateBatchSize(batchSize: int) -> int:
        if batchSize <= 0:
            raise ValueError("La dimensione del batch deve essere maggiore di zero.")
        return batchSize

    @staticmethod
    def loadBackend(modelName: str) -> SentenceTransformerBackend:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Impossibile caricare Sentence Transformers.") from error

        return SentenceTransformer(modelName)
