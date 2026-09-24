from app.models import RetrievalResult
from app.retrieval.base import Retriever
from app.retrieval.fusion import RankFusion

class HybridRetriever:
    """Combina i candidati sparse e dense tramite una strategia di fusione."""

    def __init__(
        self,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        rank_fusion: RankFusion,
        sparse_top_k: int,
        dense_top_k: int,
    ) -> None:
        self._sparse_retriever = sparse_retriever
        self._dense_retriever = dense_retriever
        self._rank_fusion = rank_fusion
        self._sparse_top_k = self._validate_top_k(sparse_top_k, "sparse_top_k")
        self._dense_top_k = self._validate_top_k(dense_top_k, "dense_top_k")

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        normalized_query = self._normalize_query(query)
        final_top_k = self._validate_top_k(k, "k")

        rankings = self._retrieve_rankings(normalized_query)
        fused_results = self._rank_fusion.fuse(rankings)
        return fused_results[:final_top_k]

    def _retrieve_rankings(self, query: str) -> list[list[RetrievalResult]]:
        sparse_results = self._sparse_retriever.retrieve(query, self._sparse_top_k)
        dense_results = self._dense_retriever.retrieve(query, self._dense_top_k)
        return [sparse_results, dense_results]

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("La query non può essere vuota.")
        return normalized_query

    @staticmethod
    def _validate_top_k(top_k: int, parameter_name: str) -> int:
        if top_k <= 0:
            raise ValueError(f"Il parametro {parameter_name} deve essere maggiore di zero.")
        return top_k