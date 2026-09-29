from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from app.indexing.sparse.base import SparseSearchMatch
from app.indexing.sparse.tokenizer import TechnicalTextTokenizer
from app.models import DocumentChunk
from app.validation import RetrievalValidator


class BM25SparseIndex:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        tokenizer: TechnicalTextTokenizer | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.tokenizer = tokenizer or TechnicalTextTokenizer()

        self.validateChunkIds()
        self.tokenizedCorpus = self.tokenizeCorpus()
        self.index = self.buildIndex()

    def getSize(self) -> int:
        return len(self.chunks)

    def getChunk(self, position: int) -> DocumentChunk:
        self.validateChunkPosition(position)
        return self.chunks[position]

    def search(self, query: str, k: int) -> list[SparseSearchMatch]:
        normalizedQuery = RetrievalValidator.normalizeQuery(query)
        RetrievalValidator.validateTopK(k, "k")

        if self.index is None:
            return []

        queryTokens = self.tokenizer.tokenize(normalizedQuery)
        if not queryTokens:
            raise ValueError("La query non contiene termini ricercabili.")

        scores = self.index.get_scores(queryTokens)
        resultCount = min(k, self.getSize())
        rankedPositions = np.argsort(-scores, kind="stable")[:resultCount]
        return self.mapSearchResults(rankedPositions, scores)

    def tokenizeCorpus(self) -> list[list[str]]:
        tokenizedCorpus: list[list[str]] = []

        for chunk in self.chunks:
            tokens = self.tokenizer.tokenize(chunk.text)
            if not tokens:
                raise ValueError(f"Il chunk '{chunk.id}' non contiene termini indicizzabili.")
            tokenizedCorpus.append(tokens)

        return tokenizedCorpus

    def buildIndex(self) -> BM25Okapi | None:
        if not self.tokenizedCorpus:
            return None
        return BM25Okapi(self.tokenizedCorpus)

    def validateChunkIds(self) -> None:
        chunkIds = [chunk.id for chunk in self.chunks]
        if len(chunkIds) != len(set(chunkIds)):
            raise ValueError("Il corpus contiene identificativi di chunk duplicati.")

    def validateChunkPosition(self, position: int) -> None:
        if position < 0 or position >= self.getSize():
            raise IndexError("La posizione richiesta non esiste nell'indice sparso.")

    @staticmethod
    def mapSearchResults(
        rankedPositions: np.ndarray,
        scores: np.ndarray,
    ) -> list[SparseSearchMatch]:
        matches: list[SparseSearchMatch] = []

        for position in rankedPositions:
            numericPosition = int(position)
            matches.append((numericPosition, float(scores[numericPosition])))

        return matches
