from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from app.models import RetrievalResult
from app.reranking.base import Reranker
from app.retrieval.base import Retriever

Clock = Callable[[], float]

@dataclass(frozen=True)
class RerankingExecution:
    """Risultati e latenze dei due stadi del retrieval con reranking."""

    results: tuple[RetrievalResult, ...]
    retrieval_latency_ms: float
    reranking_latency_ms: float

class RerankingRetriever:
    """Applica un secondo stadio di ordinamento ai candidati di un retriever."""

    def __init__(
        self,
        candidate_retriever: Retriever,
        reranker: Reranker,
        candidate_top_n: int,
        clock: Clock = perf_counter,
    ) -> None:
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._candidate_top_n = self._validate_positive_value(
            candidate_top_n,
            "candidate_top_n",
        )
        self._clock = clock

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        execution = self.retrieve_with_metrics(query, k)
        return list(execution.results)

    def retrieve_with_metrics(self, query: str, k: int) -> RerankingExecution:
        """Esegue i due stadi e restituisce le rispettive latenze."""

        normalized_query = self._normalize_query(query)
        final_top_k = self._validate_final_top_k(k)

        retrieval_start_time = self._clock()
        candidates = self._retrieve_candidates(normalized_query)
        retrieval_latency_ms = self._calculate_elapsed_time_ms(retrieval_start_time)

        if not candidates:
            return RerankingExecution(
                results=(),
                retrieval_latency_ms=retrieval_latency_ms,
                reranking_latency_ms=0.0,
            )

        reranking_start_time = self._clock()
        reranked_results = self._rerank_candidates(
            normalized_query,
            candidates,
            final_top_k,
        )
        reranking_latency_ms = self._calculate_elapsed_time_ms(reranking_start_time)

        return RerankingExecution(
            results=tuple(reranked_results[:final_top_k]),
            retrieval_latency_ms=retrieval_latency_ms,
            reranking_latency_ms=reranking_latency_ms,
        )

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
            raise ValueError(
                "Il valore Top-K finale non può superare il numero di candidati Top-N."
            )
        return validated_top_k

    @staticmethod
    def _validate_positive_value(value: int, parameter_name: str) -> int:
        if value <= 0:
            raise ValueError(f"Il parametro {parameter_name} deve essere maggiore di zero.")
        return value

    def _calculate_elapsed_time_ms(self, start_time: float) -> float:
        return (self._clock() - start_time) * 1000

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("La query non può essere vuota.")
        return normalized_query