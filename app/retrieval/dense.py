"""Dense retriever"""

from app.indexing.embeddings import EmbeddingModel
from app.indexing.vector_store import PersistentVectorIndex
from app.models import RetrievalResult

class DenseRetriever:
    """trasformazione di query -> embedding e successiva ricerca Top-K."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_index: PersistentVectorIndex,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_index = vector_index

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """Restituisce i chunk ordinati per similarità rispetto alla query."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("La query non può essere vuota.")
        if k <= 0:
            raise ValueError("Il numero di risultati deve essere maggiore di zero.")
        if self._vector_index.size == 0:
            return []

        query_vectors = self._embedding_model.encode([normalized_query])
        if len(query_vectors) != 1:
            raise ValueError("Il modello deve restituire un solo embedding per la query.")

        matches = self._vector_index.search(query_vectors[0], k)
        return [
            RetrievalResult(
                chunk=self._vector_index.get_chunk(position),
                rank=rank,
                score=score,
                retriever="dense",
            )
            for rank, (position, score) in enumerate(matches, start=1)
        ]