from collections.abc import Sequence
from typing import Protocol
from app.indexing.embeddings.base import EmbeddingVector

#Questa classe fa da passacarte verso la libreria di sentence transformation per adoperare la vettorizzazione dle testo

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
        
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("model_name non può essere vuoto")
        if batch_size <= 0:
            raise ValueError("batch_size deve essere maggiore di zero")

        self.model_name = normalized_model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._backend = (
            backend if backend is not None else self._load_backend(normalized_model_name)
        )
    
    def encode(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        text_batch = list(texts)
        if not text_batch:
            return []

        encoded_vectors = self._backend.encode(
            text_batch,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        vectors = [[float(value) for value in vector] for vector in encoded_vectors]

        if len(vectors) != len(text_batch):
            raise ValueError("Errore nella restituzione del testo")

        return vectors

    #Carico il modello
    @staticmethod
    def _load_backend(model_name: str) -> SentenceTransformerBackend:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Errore nel caricamento del modello"
            ) from exc

        return SentenceTransformer(model_name)