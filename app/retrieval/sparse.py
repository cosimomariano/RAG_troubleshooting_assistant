from app.indexing.sparse import SparseIndex, SparseSearchMatch
from app.models import RetrievalResult


class SparseRetriever:
    RETRIEVER_NAME = "sparse"

    def __init__(self, sparseIndex: SparseIndex) -> None:
        self.sparseIndex = sparseIndex

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        sparseMatches = self.sparseIndex.search(query, k)
        return self.mapResults(sparseMatches)

    def mapResults(
        self,
        sparseMatches: list[SparseSearchMatch],
    ) -> list[RetrievalResult]:
        retrievalResults: list[RetrievalResult] = []

        for rank, sparseMatch in enumerate(sparseMatches, start=1):
            position, score = sparseMatch
            retrievalResults.append(
                RetrievalResult(
                    chunk=self.sparseIndex.getChunk(position),
                    rank=rank,
                    score=score,
                    retriever=self.RETRIEVER_NAME,
                )
            )

        return retrievalResults
