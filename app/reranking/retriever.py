from collections.abc import Callable, Sequence
from time import perf_counter

from app.models import RetrievalResult
from app.reranking.base import Reranker
from app.retrieval.base import RetrievalExecution, Retriever
from app.validation import RetrievalValidator

Clock = Callable[[], float]


class RerankingRetriever:
    """Applica un secondo stadio di ordinamento ai candidati di un retriever."""

    def __init__(
        self,
        candidateRetriever: Retriever,
        reranker: Reranker,
        candidateTopN: int,
        clock: Clock = perf_counter,
    ) -> None:
        self.candidateRetriever = candidateRetriever
        self.reranker = reranker
        self.candidateTopN = RetrievalValidator.validateTopK(
            candidateTopN,
            "candidate_top_n",
        )
        self.clock = clock

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        execution = self.retrieveWithMetrics(query, k)
        return list(execution.results)

    def retrieveWithMetrics(self, query: str, k: int) -> RetrievalExecution:
        """Esegue i due stadi e restituisce le rispettive latenze."""

        normalizedQuery = RetrievalValidator.normalizeQuery(query)
        finalTopK = self.validateFinalTopK(k)

        retrievalStartTime = self.clock()
        candidates = self.retrieveCandidates(normalizedQuery)
        retrievalLatencyMs = self.calculateElapsedTimeMs(retrievalStartTime)

        if not candidates:
            return RetrievalExecution(
                results=(),
                retrievalLatencyMs=retrievalLatencyMs,
                rerankingLatencyMs=0.0,
            )

        rerankingStartTime = self.clock()
        rerankedResults = self.rerankCandidates(
            normalizedQuery,
            candidates,
            finalTopK,
        )
        rerankingLatencyMs = self.calculateElapsedTimeMs(rerankingStartTime)

        return RetrievalExecution(
            results=tuple(rerankedResults[:finalTopK]),
            retrievalLatencyMs=retrievalLatencyMs,
            rerankingLatencyMs=rerankingLatencyMs,
        )

    def retrieveCandidates(self, query: str) -> list[RetrievalResult]:
        return self.candidateRetriever.retrieve(query, self.candidateTopN)

    def rerankCandidates(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        finalTopK: int,
    ) -> list[RetrievalResult]:
        return self.reranker.rerank(query, candidates, finalTopK)

    def validateFinalTopK(self, finalTopK: int) -> int:
        validatedTopK = RetrievalValidator.validateTopK(finalTopK, "k")
        if validatedTopK > self.candidateTopN:
            raise ValueError(
                "Il valore Top-K finale non può superare il numero di candidati Top-N."
            )
        return validatedTopK

    def calculateElapsedTimeMs(self, startTime: float) -> float:
        return (self.clock() - startTime) * 1000
