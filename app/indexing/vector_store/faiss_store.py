import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.indexing.embeddings import EmbeddingVector
from app.indexing.vector_store.base import VectorSearchMatch
from app.models import DocumentChunk
from app.validation import RetrievalValidator

_INDEX_FILE_NAME = "dense.index"
_CHUNKS_FILE_NAME = "chunks.json"
_SCHEMA_VERSION = 1
_METRIC_NAME = "inner_product"


class FaissVectorIndex:
    def __init__(self, dimension: int) -> None:
        self.validateDimension(dimension)
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[DocumentChunk] = []

    def getDimension(self) -> int:
        return int(self.index.d)

    def getSize(self) -> int:
        return int(self.index.ntotal)

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[EmbeddingVector],
    ) -> None:
        chunkBatch = list(chunks)
        vectorBatch = list(vectors)

        self.validateBatchSizes(chunkBatch, vectorBatch)
        if not chunkBatch:
            return

        self.validateChunkIds(chunkBatch)
        vectorMatrix = self.createVectorMatrix(vectorBatch)
        self.validateVectorMatrix(vectorMatrix)

        self.index.add(np.ascontiguousarray(vectorMatrix))
        self.chunks.extend(chunkBatch)
        self.validateAlignment()

    def getChunk(self, position: int) -> DocumentChunk:
        self.validateChunkPosition(position)
        return self.chunks[position]

    def search(self, vector: EmbeddingVector, k: int) -> list[VectorSearchMatch]:
        RetrievalValidator.validateTopK(k, "k")
        if self.getSize() == 0:
            return []

        queryMatrix = self.createQueryMatrix(vector)
        resultCount = min(k, self.getSize())
        scores, positions = self.index.search(queryMatrix, resultCount)
        return self.mapSearchResults(positions[0], scores[0])

    def save(self, directoryPath: Path) -> None:
        self.validateAlignment()
        directoryPath.mkdir(parents=True, exist_ok=True)

        indexPath, chunksPath = self.buildStoragePaths(directoryPath)
        faiss.write_index(self.index, str(indexPath))
        chunksPath.write_text(
            json.dumps(self.buildPayload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directoryPath: Path) -> "FaissVectorIndex":
        indexPath, chunksPath = cls.buildStoragePaths(directoryPath)
        cls.validateStorageFiles(indexPath, chunksPath)

        storedIndex = faiss.read_index(str(indexPath))
        payload = cls.readPayload(chunksPath)
        cls.validatePayload(payload, int(storedIndex.d))

        instance = cls(dimension=int(storedIndex.d))
        instance.index = storedIndex
        instance.chunks = cls.deserializeChunks(payload)
        instance.validateAlignment()
        return instance

    def buildPayload(self) -> dict[str, Any]:
        serializedChunks: list[dict[str, Any]] = []
        for chunk in self.chunks:
            serializedChunks.append(chunk.model_dump(mode="json"))

        return {
            "schema_version": _SCHEMA_VERSION,
            "metric": _METRIC_NAME,
            "dimension": self.getDimension(),
            "chunks": serializedChunks,
        }

    def validateChunkIds(self, chunks: list[DocumentChunk]) -> None:
        newChunkIds = [chunk.id for chunk in chunks]
        if len(newChunkIds) != len(set(newChunkIds)):
            raise ValueError("Il batch contiene identificativi di chunk duplicati.")

        storedChunkIds = {chunk.id for chunk in self.chunks}
        if storedChunkIds.intersection(newChunkIds):
            raise ValueError("Uno o più chunk sono già presenti nell'indice.")

    def validateVectorMatrix(self, vectorMatrix: np.ndarray) -> None:
        if vectorMatrix.ndim != 2 or vectorMatrix.shape[1] != self.getDimension():
            raise ValueError(
                f"Ogni vettore deve avere dimensione {self.getDimension()}; "
                f"forma ricevuta: {vectorMatrix.shape}."
            )

    def validateChunkPosition(self, position: int) -> None:
        if position < 0 or position >= self.getSize():
            raise IndexError("La posizione richiesta non esiste nell'indice.")

    def validateAlignment(self) -> None:
        if self.getSize() != len(self.chunks):
            raise ValueError("L'indice FAISS e la mappatura dei chunk contengono quantità diverse.")

    @staticmethod
    def createVectorMatrix(vectors: list[EmbeddingVector]) -> np.ndarray:
        try:
            return np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("I vettori devono formare una matrice numerica regolare.") from error

    def createQueryMatrix(self, vector: EmbeddingVector) -> np.ndarray:
        try:
            queryMatrix = np.asarray([vector], dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("Il vettore di ricerca deve contenere valori numerici.") from error

        if queryMatrix.ndim != 2 or queryMatrix.shape[1] != self.getDimension():
            raise ValueError(
                f"Il vettore di ricerca deve avere dimensione {self.getDimension()}; "
                f"forma ricevuta: {queryMatrix.shape}."
            )
        if not np.isfinite(queryMatrix).all():
            raise ValueError("Il vettore di ricerca contiene valori non finiti.")
        return np.ascontiguousarray(queryMatrix)

    @staticmethod
    def mapSearchResults(
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
    def validateDimension(dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("La dimensione dei vettori deve essere maggiore di zero.")

    @staticmethod
    def validateBatchSizes(
        chunks: list[DocumentChunk],
        vectors: list[EmbeddingVector],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Il numero di chunk deve coincidere con quello dei vettori.")

    @staticmethod
    def buildStoragePaths(directoryPath: Path) -> tuple[Path, Path]:
        return (
            directoryPath / _INDEX_FILE_NAME,
            directoryPath / _CHUNKS_FILE_NAME,
        )

    @staticmethod
    def validateStorageFiles(indexPath: Path, chunksPath: Path) -> None:
        if not indexPath.is_file() or not chunksPath.is_file():
            raise FileNotFoundError(
                "La directory non contiene un indice FAISS e la relativa mappatura."
            )

    @staticmethod
    def readPayload(chunksPath: Path) -> dict[str, Any]:
        try:
            payload = json.loads(chunksPath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("La mappatura dei chunk non contiene JSON valido.") from error

        if not isinstance(payload, dict):
            raise ValueError("La mappatura dei chunk persistita non è valida.")
        return payload

    @staticmethod
    def validatePayload(payload: dict[str, Any], indexDimension: int) -> None:
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("La versione del formato persistito non è supportata.")
        if payload.get("metric") != _METRIC_NAME:
            raise ValueError("La metrica dell'indice persistito non è supportata.")
        if payload.get("dimension") != indexDimension:
            raise ValueError("La dimensione salvata non coincide con quella dell'indice FAISS.")
        if not isinstance(payload.get("chunks"), list):
            raise ValueError("La mappatura dei chunk persistita non è valida.")

    @staticmethod
    def deserializeChunks(payload: dict[str, Any]) -> list[DocumentChunk]:
        rawChunks = payload["chunks"]
        return [DocumentChunk.model_validate(rawChunk) for rawChunk in rawChunks]
