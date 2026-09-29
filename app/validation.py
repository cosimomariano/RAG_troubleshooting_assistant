class RetrievalValidator:
    """Raccoglie le regole comuni agli stadi di retrieval e reranking."""

    @staticmethod
    def normalizeQuery(query: str) -> str:
        normalizedQuery = query.strip()
        if not normalizedQuery:
            raise ValueError("La query non può essere vuota.")
        return normalizedQuery

    @staticmethod
    def validateTopK(topK: int, parameterName: str = "Top-K") -> int:
        if topK <= 0:
            raise ValueError(f"Il parametro {parameterName} deve essere maggiore di zero.")
        return topK
