from app.models import RetrievalResult
from app.retrieval.base import Retriever
from app.retrieval.fusion import RankFusion
from app.validation import RetrievalValidator


class HybridRetriever:
    """Combina i candidati sparse e dense tramite una strategia di fusione."""

    def __init__(
        self,
        sparseRetriever: Retriever,
        denseRetriever: Retriever,
        rankFusion: RankFusion,
        sparseTopK: int,
        denseTopK: int,
    ) -> None:
        self.sparseRetriever = sparseRetriever
        self.denseRetriever = denseRetriever
        self.rankFusion = rankFusion
        self.sparseTopK = RetrievalValidator.validateTopK(
            sparseTopK,
            "sparse_top_k",
        )
        self.denseTopK = RetrievalValidator.validateTopK(
            denseTopK,
            "dense_top_k",
        )

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        normalizedQuery = RetrievalValidator.normalizeQuery(query)
        finalTopK = RetrievalValidator.validateTopK(k, "k")

        rankings = self.retrieveRankings(normalizedQuery)
        fusedResults = self.rankFusion.fuse(rankings)
        return fusedResults[:finalTopK]

    def retrieveRankings(self, query: str) -> list[list[RetrievalResult]]:
        sparseResults = self.sparseRetriever.retrieve(query, self.sparseTopK)
        denseResults = self.denseRetriever.retrieve(query, self.denseTopK)
        return [sparseResults, denseResults]
