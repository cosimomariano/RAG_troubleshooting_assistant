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
    def __init__(self, dimension: int) -> None:
        self._validate_dimension(dimension)
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[DocumentChunk] = []

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[EmbeddingVector],
    ) -> None:
        chunk_batch = list(chunks)
        vector_batch = list(vectors)

        self._validate_batch_sizes(chunk_batch, vector_batch)
        if not chunk_batch:
            return

        self._validate_chunk_ids(chunk_batch)
        vector_matrix = self._create_vector_matrix(vector_batch)
        self._validate_vector_matrix(vector_matrix)

        self._index.add(np.ascontiguousarray(vector_matrix))
        self._chunks.extend(chunk_batch)
        self._validate_alignment()

    def get_chunk(self, position: int) -> DocumentChunk:
        self._validate_chunk_position(position)
        return self._chunks[position]

    def search(self, vector: EmbeddingVector, k: int) -> list[VectorSearchMatch]:
        self._validate_result_count(k)
        if self.size == 0:
            return []

        query_matrix = self._create_query_matrix(vector)
        result_count = min(k, self.size)
        scores, positions = self._index.search(query_matrix, result_count)
        return self._map_search_results(positions[0], scores[0])

    def save(self, directory_path: Path) -> None:
        self._validate_alignment()
        directory_path.mkdir(parents=True, exist_ok=True)

        index_path, chunks_path = self._build_storage_paths(directory_path)
        faiss.write_index(self._index, str(index_path))
        chunks_path.write_text(
            json.dumps(self._build_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory_path: Path) -> "FaissVectorIndex":
        index_path, chunks_path = cls._build_storage_paths(directory_path)
        cls._validate_storage_files(index_path, chunks_path)

        stored_index = faiss.read_index(str(index_path))
        payload = cls._read_payload(chunks_path)
        cls._validate_payload(payload, int(stored_index.d))

        instance = cls(dimension=int(stored_index.d))
        instance._index = stored_index
        instance._chunks = cls._deserialize_chunks(payload)
        instance._validate_alignment()
        return instance

    def _build_payload(self) -> dict[str, Any]:
        serialized_chunks: list[dict[str, Any]] = []
        for chunk in self._chunks:
            serialized_chunks.append(chunk.model_dump(mode="json"))

        return {
            "schema_version": _SCHEMA_VERSION,
            "metric": _METRIC_NAME,
            "dimension": self.dimension,
            "chunks": serialized_chunks,
        }

    def _validate_chunk_ids(self, chunks: list[DocumentChunk]) -> None:
        new_chunk_ids = [chunk.id for chunk in chunks]
        if len(new_chunk_ids) != len(set(new_chunk_ids)):
            raise ValueError("Il batch contiene identificativi di chunk duplicati.")

        stored_chunk_ids = {chunk.id for chunk in self._chunks}
        if stored_chunk_ids.intersection(new_chunk_ids):
            raise ValueError("Uno o più chunk sono già presenti nell'indice.")

    def _validate_vector_matrix(self, vector_matrix: np.ndarray) -> None:
        if vector_matrix.ndim != 2 or vector_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Ogni vettore deve avere dimensione {self.dimension}; "
                f"forma ricevuta: {vector_matrix.shape}."
            )

    def _validate_chunk_position(self, position: int) -> None:
        if position < 0 or position >= self.size:
            raise IndexError("La posizione richiesta non esiste nell'indice.")

    def _validate_alignment(self) -> None:
        if self.size != len(self._chunks):
            raise ValueError("L'indice FAISS e la mappatura dei chunk contengono quantità diverse.")

    @staticmethod
    def _create_vector_matrix(vectors: list[EmbeddingVector]) -> np.ndarray:
        try:
            return np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("I vettori devono formare una matrice numerica regolare.") from error

    def _create_query_matrix(self, vector: EmbeddingVector) -> np.ndarray:
        try:
            query_matrix = np.asarray([vector], dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Il vettore di ricerca deve contenere valori numerici.") from error

        if query_matrix.ndim != 2 or query_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Il vettore di ricerca deve avere dimensione {self.dimension}; "
                f"forma ricevuta: {query_matrix.shape}."
            )
        if not np.isfinite(query_matrix).all():
            raise ValueError("Il vettore di ricerca contiene valori non finiti.")
        return np.ascontiguousarray(query_matrix)

    @staticmethod
    def _map_search_results(
        positions: np.ndarray,
        scores: np.ndarray,
    ) -> list[VectorSearchMatch]:
        matches: list[VectorSearchMatch] = []
        for position, score in zip(positions, scores, strict=True):
            if position < 0:
                continue
            matches.append((int(position), float(score)))
        return matches

    @staticmethod
    def _validate_dimension(dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("La dimensione dei vettori deve essere maggiore di zero.")

    @staticmethod
    def _validate_batch_sizes(
        chunks: list[DocumentChunk],
        vectors: list[EmbeddingVector],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Il numero di chunk deve coincidere con quello dei vettori.")

    @staticmethod
    def _validate_result_count(result_count: int) -> None:
        if result_count <= 0:
            raise ValueError("Il numero di risultati deve essere maggiore di zero.")

    @staticmethod
    def _build_storage_paths(directory_path: Path) -> tuple[Path, Path]:
        return (
            directory_path / _INDEX_FILE_NAME,
            directory_path / _CHUNKS_FILE_NAME,
        )

    @staticmethod
    def _validate_storage_files(index_path: Path, chunks_path: Path) -> None:
        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                "La directory non contiene un indice FAISS e la relativa mappatura."
            )

    @staticmethod
    def _read_payload(chunks_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("La mappatura dei chunk non contiene JSON valido.") from error

        if not isinstance(payload, dict):
            raise ValueError("La mappatura dei chunk persistita non è valida.")
        return payload

    @staticmethod
    def _validate_payload(payload: dict[str, Any], index_dimension: int) -> None:
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("La versione del formato persistito non è supportata.")
        if payload.get("metric") != _METRIC_NAME:
            raise ValueError("La metrica dell'indice persistito non è supportata.")
        if payload.get("dimension") != index_dimension:
            raise ValueError("La dimensione salvata non coincide con quella dell'indice FAISS.")
        if not isinstance(payload.get("chunks"), list):
            raise ValueError("La mappatura dei chunk persistita non è valida.")

    @staticmethod
    def _deserialize_chunks(payload: dict[str, Any]) -> list[DocumentChunk]:
        raw_chunks = payload["chunks"]
        return [DocumentChunk.model_validate(raw_chunk) for raw_chunk in raw_chunks]
