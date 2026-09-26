from collections.abc import Collection, Sequence

class RetrievalMetricsCalculator:
    """Calcola metriche di retrieval a partire da graduatorie di documenti."""

    @staticmethod
    def recall_at_k(
        retrieved_documents: Sequence[str],
        relevant_documents: Collection[str],
        k: int,
    ) -> float:
        """Restituisce la quota di documenti rilevanti recuperati nelle prime k posizioni."""
        RetrievalMetricsCalculator._validate_k(k)
        relevant_document_set = RetrievalMetricsCalculator._to_document_set(
            relevant_documents
        )

        retrieved_document_set = set(retrieved_documents[:k])
        retrieved_relevant_documents = retrieved_document_set.intersection(
            relevant_document_set
        )

        return len(retrieved_relevant_documents) / len(relevant_document_set)

    @staticmethod
    def reciprocal_rank(
        retrieved_documents: Sequence[str],
        relevant_documents: Collection[str],
    ) -> float:
        """Restituisce l'inverso della posizione del primo documento rilevante."""
        relevant_document_set = RetrievalMetricsCalculator._to_document_set(
            relevant_documents
        )

        for rank, document in enumerate(retrieved_documents, start=1):
            if document in relevant_document_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_rankings: Sequence[Sequence[str]],
        relevant_documents_by_case: Sequence[Collection[str]],
    ) -> float:
        """Calcola la media dei reciprocal rank di tutti i casi sperimentali."""
        if not retrieved_rankings:
            raise ValueError("È richiesta almeno una graduatoria di documenti.")

        if len(retrieved_rankings) != len(relevant_documents_by_case):
            raise ValueError(
                "Il numero di graduatorie deve coincidere con il numero dei casi."
            )

        reciprocal_rank_sum = 0.0
        for retrieved_documents, relevant_documents in zip(
            retrieved_rankings,
            relevant_documents_by_case,
            strict=True,
        ):
            reciprocal_rank_sum += RetrievalMetricsCalculator.reciprocal_rank(
                retrieved_documents,
                relevant_documents,
            )

        return reciprocal_rank_sum / len(retrieved_rankings)

    @staticmethod
    def _validate_k(k: int) -> None:
        if k < 1:
            raise ValueError("Il valore di k deve essere maggiore di zero.")

    @staticmethod
    def _to_document_set(documents: Collection[str]) -> set[str]:
        document_set = set(documents)
        if not document_set:
            raise ValueError("È richiesto almeno un documento rilevante.")
        return document_set