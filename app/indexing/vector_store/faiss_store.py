"""Implementazione persistente del vector store basata su FAISS."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import faiss
import numpy as np
from app.indexing.embeddings import EmbeddingVector
from app.indexing.vector_store.base import VectorSearchMatch
from app.models import DocumentChunk

_INDEX_FILE_NAME = "dense.index"
_CHUNKS_FILE_NAME = "chunks.json"
_SCHEMA_VERSION = 1
_METRIC_NAME = "inner_product"

class FaissVectorIndex:
    """Conserva vettori FAISS e chunk nella stessa posizione logica."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("La dimensione dei vettori deve essere maggiore di zero.")

        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[DocumentChunk] = []

    @property
    def dimension(self) -> int:
        """Restituisce la dimensione attesa per ogni vettore."""

        return int(self._index.d)

    @property
    def size(self) -> int:
        """Restituisce il numero di elementi presenti nell'indice."""

        return int(self._index.ntotal)

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[EmbeddingVector],
    ) -> None:
        """Aggiunge chunk e vettori mantenendo allineate le loro posizioni."""

        chunk_batch = list(chunks)
        vector_batch = list(vectors)

        if len(chunk_batch) != len(vector_batch):
            raise ValueError("Il numero di chunk deve coincidere con quello dei vettori.")
        if not chunk_batch:
            return

        new_ids = [chunk.id for chunk in chunk_batch]
        if len(new_ids) != len(set(new_ids)):
            raise ValueError("Il batch contiene identificativi di chunk duplicati.")

        stored_ids = {chunk.id for chunk in self._chunks}
        if stored_ids.intersection(new_ids):
            raise ValueError("Uno o più chunk sono già presenti nell'indice.")

        try:
            matrix = np.asarray(vector_batch, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("I vettori devono formare una matrice numerica regolare.") from exc

        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Ogni vettore deve avere dimensione {self.dimension}; "
                f"forma ricevuta: {matrix.shape}."
            )

        self._index.add(np.ascontiguousarray(matrix))
        self._chunks.extend(chunk_batch)
        self._ensure_alignment()

    def get_chunk(self, position: int) -> DocumentChunk:
        """Recupera il chunk associato all'indice."""

        if position < 0 or position >= self.size:
            raise IndexError("La posizione richiesta non esiste nell'indice.")
        return self._chunks[position]

    def search(self, vector: EmbeddingVector, k: int) -> list[VectorSearchMatch]:
        """Calcolo posizione e score dei vectors piu vicini"""

        if k <= 0:
            raise ValueError("Il numero di risultati deve essere maggiore di zero.")
        if self.size == 0:
            return []

        try:
            query_matrix = np.asarray([vector], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Il vettore di ricerca deve contenere valori numerici.") from exc

        if query_matrix.ndim != 2 or query_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Il vettore di ricerca deve avere dimensione {self.dimension}; "
                f"forma ricevuta: {query_matrix.shape}."
            )
        if not np.isfinite(query_matrix).all():
            raise ValueError("Il vettore di ricerca contiene valori non finiti.")

        result_count = min(k, self.size)
        scores, positions = self._index.search(
            np.ascontiguousarray(query_matrix),
            result_count,
        )
        return [
            (int(position), float(score))
            for position, score in zip(positions[0], scores[0], strict=True)
            if position >= 0
        ]

    def save(self, directory_path: Path) -> None:
        """Salva un indice sullo store"""

        self._ensure_alignment()
        directory_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory_path / _INDEX_FILE_NAME))
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "metric": _METRIC_NAME,
            "dimension": self.dimension,
            "chunks": [chunk.model_dump(mode="json") for chunk in self._chunks],
        }
        (directory_path / _CHUNKS_FILE_NAME).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory_path: Path) -> "FaissVectorIndex":
        """Carica un indice dallo store"""

        index_path = directory_path / _INDEX_FILE_NAME
        chunks_path = directory_path / _CHUNKS_FILE_NAME
        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                "La directory non contiene un indice FAISS e la relativa mappatura."
            )

        index = faiss.read_index(str(index_path))
        payload = cls._read_payload(chunks_path)

        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("La versione del formato persistito non è supportata.")
        if payload.get("metric") != _METRIC_NAME:
            raise ValueError("La metrica dell'indice persistito non è supportata.")
        if payload.get("dimension") != int(index.d):
            raise ValueError("La dimensione salvata non coincide con quella dell'indice FAISS.")

        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError("La mappatura dei chunk persistita non è valida.")

        instance = cls(dimension=int(index.d))
        instance._index = index
        instance._chunks = [DocumentChunk.model_validate(chunk) for chunk in raw_chunks]
        instance._ensure_alignment()
        return instance

    @staticmethod
    def _read_payload(chunks_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("La mappatura dei chunk non contiene JSON valido.") from exc

        if not isinstance(payload, dict):
            raise ValueError("La mappatura dei chunk persistita non è valida.")
        return payload

    def _ensure_alignment(self) -> None:
        if self.size != len(self._chunks):
            raise ValueError("L'indice FAISS e la mappatura dei chunk contengono quantità diverse.")