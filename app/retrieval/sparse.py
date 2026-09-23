from app.indexing.sparse import SparseIndex, SparseSearchMatch
from app.models import RetrievalResult


class SparseRetriever:
    RETRIEVER_NAME = "sparse"

    def __init__(self, sparse_index: SparseIndex) -> None:
        self._sparse_index = sparse_index

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        sparse_matches = self._sparse_index.search(query, k)
        return self._map_results(sparse_matches)

    def _map_results(
        self,
        sparse_matches: list[SparseSearchMatch],
    ) -> list[RetrievalResult]:
        retrieval_results: list[RetrievalResult] = []

        for rank, sparse_match in enumerate(sparse_matches, start=1):
            position, score = sparse_match
            retrieval_results.append(
                RetrievalResult(
                    chunk=self._sparse_index.get_chunk(position),
                    rank=rank,
                    score=score,
                    retriever=self.RETRIEVER_NAME,
                )
            )

        return retrieval_results
