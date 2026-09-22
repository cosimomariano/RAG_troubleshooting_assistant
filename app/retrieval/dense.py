from app.indexing.embeddings import EmbeddingModel, EmbeddingVector
from app.indexing.vector_store import PersistentVectorIndex, VectorSearchMatch
from app.models import RetrievalResult


class DenseRetriever:
    RETRIEVER_NAME = "dense"

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: PersistentVectorIndex,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_index = vector_index

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        normalized_query = self._normalize_query(query)
        self._validate_result_count(k)

        if self._vector_index.size == 0:
            return []

        query_vector = self._encode_query(normalized_query)
        vector_matches = self._vector_index.search(query_vector, k)
        return self._map_results(vector_matches)

    def _encode_query(self, query: str) -> EmbeddingVector:
        query_vectors = self._embedding_model.encode([query])
        if len(query_vectors) != 1:
            raise ValueError("Il modello deve restituire un solo embedding per la query.")
        return query_vectors[0]

    def _map_results(
        self,
        vector_matches: list[VectorSearchMatch],
    ) -> list[RetrievalResult]:
        retrieval_results: list[RetrievalResult] = []

        for rank, vector_match in enumerate(vector_matches, start=1):
            position, score = vector_match
            retrieval_result = RetrievalResult(
                chunk=self._vector_index.get_chunk(position),
                rank=rank,
                score=score,
                retriever=self.RETRIEVER_NAME,
            )
            retrieval_results.append(retrieval_result)

        return retrieval_results

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
