from collections.abc import Sequence
import numpy as np
from rank_bm25 import BM25Okapi
from app.indexing.sparse.base import SparseSearchMatch
from app.indexing.sparse.tokenizer import TechnicalTextTokenizer
from app.models import DocumentChunk

class BM25SparseIndex:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        tokenizer: TechnicalTextTokenizer | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._tokenizer = tokenizer or TechnicalTextTokenizer()

        self._validate_chunk_ids()
        self._tokenized_corpus = self._tokenize_corpus()
        self._index = self._build_index()

    @property
    def size(self) -> int:
        return len(self._chunks)

    # Indicizzazione dei chunk
    def get_chunk(self, position: int) -> DocumentChunk:
        self._validate_chunk_position(position)
        return self._chunks[position]

    def search(self, query: str, k: int) -> list[SparseSearchMatch]:
        normalized_query = self._normalize_query(query)
        self._validate_result_count(k)

        if self._index is None:
            return []

        query_tokens = self._tokenizer.tokenize(normalized_query)
        if not query_tokens:
            raise ValueError("La query non contiene termini ricercabili.")

        scores = self._index.get_scores(query_tokens)
        result_count = min(k, self.size)
        ranked_positions = np.argsort(-scores, kind="stable")[:result_count]
        return self._map_search_results(ranked_positions, scores)

    # Assegnazione dei token tramite BM25 OKapi
    def _tokenize_corpus(self) -> list[list[str]]:
        tokenized_corpus: list[list[str]] = []

        for chunk in self._chunks:
            tokens = self._tokenizer.tokenize(chunk.text)
            if not tokens:
                raise ValueError(f"Il chunk '{chunk.id}' non contiene termini indicizzabili.")
            tokenized_corpus.append(tokens)

        return tokenized_corpus

    def _build_index(self) -> BM25Okapi | None:
        if not self._tokenized_corpus:
            return None
        return BM25Okapi(self._tokenized_corpus)


   # Metodi di validazione

    def _validate_chunk_ids(self) -> None:
        chunk_ids = [chunk.id for chunk in self._chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Il corpus contiene identificativi di chunk duplicati.")

    def _validate_chunk_position(self, position: int) -> None:
        if position < 0 or position >= self.size:
            raise IndexError("La posizione richiesta non esiste nell'indice sparso.")

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("La query non può essere vuota.")
        return normalized_query

    @staticmethod
    def _validate_result_count(result_count: int) -> None:
        if result_count <= 0:
            raise ValueError("Il numero di risultati deve essere maggiore di zero.")

    @staticmethod
    def _map_search_results(
        ranked_positions: np.ndarray,
        scores: np.ndarray,
    ) -> list[SparseSearchMatch]:
        matches: list[SparseSearchMatch] = []

        for position in ranked_positions:
            numeric_position = int(position)
            matches.append((numeric_position, float(scores[numeric_position])))

        return matches
