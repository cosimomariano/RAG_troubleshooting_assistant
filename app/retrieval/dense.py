from app.indexing.embeddings import EmbeddingModel, EmbeddingVector
from app.indexing.vector_store import PersistentVectorIndex, VectorSearchMatch
from app.models import RetrievalResult
from app.validation import RetrievalValidator


class DenseRetriever:
    RETRIEVER_NAME = "dense"

    def __init__(
        self,
        embeddingModel: EmbeddingModel,
        vectorIndex: PersistentVectorIndex,
    ) -> None:
        self.embeddingModel = embeddingModel
        self.vectorIndex = vectorIndex

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        normalizedQuery = RetrievalValidator.normalizeQuery(query)
        RetrievalValidator.validateTopK(k, "k")

        if self.vectorIndex.getSize() == 0:
            return []

        queryVector = self.encodeQuery(normalizedQuery)
        vectorMatches = self.vectorIndex.search(queryVector, k)
        return self.mapResults(vectorMatches)

    def encodeQuery(self, query: str) -> EmbeddingVector:
        queryVectors = self.embeddingModel.encode([query])
        if len(queryVectors) != 1:
            raise ValueError("Il modello deve restituire un solo embedding per la query.")
        return queryVectors[0]

    def mapResults(
        self,
        vectorMatches: list[VectorSearchMatch],
    ) -> list[RetrievalResult]:
        retrievalResults: list[RetrievalResult] = []

        for rank, vectorMatch in enumerate(vectorMatches, start=1):
            position, score = vectorMatch
            retrievalResult = RetrievalResult(
                chunk=self.vectorIndex.getChunk(position),
                rank=rank,
                score=score,
                retriever=self.RETRIEVER_NAME,
            )
            retrievalResults.append(retrievalResult)

        return retrievalResults
