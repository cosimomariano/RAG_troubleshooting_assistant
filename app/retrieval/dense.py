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
        # Normalizzazione e prevalidazione
        normalizedQuery = RetrievalValidator.normalizeQuery(query)
        RetrievalValidator.validateTopK(k, "k")

        if self.vectorIndex.getSize() == 0:
            return []

        # Encoding della query normalizzata
        queryVector = self.encodeQuery(normalizedQuery)

        # Ricerca del vettore nel database vettoriale
        vectorMatches = self.vectorIndex.search(queryVector, k)

        # Mapping dei risultati con i vettori recuperati
        return self.mapResults(vectorMatches)

    def encodeQuery(self, query: str) -> EmbeddingVector:
        # Utilizzo del modello di embeddding fornito in configurazione per l'encoding della query
        queryVectors = self.embeddingModel.encode([query])
        if len(queryVectors) != 1:
            raise ValueError("Il modello deve restituire un solo embedding per la query.")
        return queryVectors[0]

    def mapResults(
        self,
        vectorMatches: list[VectorSearchMatch],
    ) -> list[RetrievalResult]:
        retrievalResults: list[RetrievalResult] = []

        # Per ogni vettore recuperato incapsulo i dati in un oggetto RetrievalResult 
        # utile per le lavorazioni successive
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
