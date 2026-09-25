from collections.abc import Sequence
from app.models import RetrievalResult
from app.reranking.base import Reranker
from app.retrieval.base import Retriever

class RerankingRetriever:
    """Applica un secondo stadio di ordinamento ai candidati di un retriever."""

    def __init__(
        self,
        candidate_retriever: Retriever,
        reranker: Reranker,
        candidate_top_n: int,
    ) -> None:
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._candidate_top_n = self._validate_positive_value(
            candidate_top_n,
            "candidate_top_n",
        )

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        normalized_query = self._normalize_query(query)
        final_top_k = self._validate_final_top_k(k)
        candidates = self._retrieve_candidates(normalized_query)

        if not candidates:
            return []

        reranked_results = self._rerank_candidates(
            normalized_query,
            candidates,
            final_top_k,
        )
        return reranked_results[:final_top_k]

    def _retrieve_candidates(self, query: str) -> list[RetrievalResult]:
        return self._candidate_retriever.retrieve(query, self._candidate_top_n)

    def _rerank_candidates(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        final_top_k: int,
    ) -> list[RetrievalResult]:
        return self._reranker.rerank(query, candidates, final_top_k)

    def _validate_final_top_k(self, final_top_k: int) -> int:
        validated_top_k = self._validate_positive_value(final_top_k, "k")
        if validated_top_k > self._candidate_top_n:
            raise ValueError("Il valore Top-K finale non può superare il numero di candidati Top-N.")
        return validated_top_k

    @staticmethod
    def _validate_positive_value(value: int, parameter_name: str) -> int:
        if value <= 0:
            raise ValueError(f"Il parametro {parameter_name} deve essere maggiore di zero.")
        return value

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("La query non può essere vuota.")
        return normalized_query