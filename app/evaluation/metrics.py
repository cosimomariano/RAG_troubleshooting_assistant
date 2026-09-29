from collections.abc import Collection, Sequence


class RetrievalMetricsCalculator:
    """Calcola metriche di retrieval a partire da graduatorie di documenti."""

    @staticmethod
    def recallAtK(
        retrievedDocuments: Sequence[str],
        relevantDocuments: Collection[str],
        k: int,
    ) -> float:
        """Restituisce la quota di documenti rilevanti recuperati nelle prime k posizioni."""
        RetrievalMetricsCalculator.validateK(k)
        relevantDocumentSet = RetrievalMetricsCalculator.toDocumentSet(relevantDocuments)

        retrievedDocumentSet = set(retrievedDocuments[:k])
        retrievedRelevantDocuments = retrievedDocumentSet.intersection(relevantDocumentSet)

        return len(retrievedRelevantDocuments) / len(relevantDocumentSet)

    @staticmethod
    def reciprocalRank(
        retrievedDocuments: Sequence[str],
        relevantDocuments: Collection[str],
    ) -> float:
        """Restituisce l'inverso della posizione del primo documento rilevante."""
        relevantDocumentSet = RetrievalMetricsCalculator.toDocumentSet(relevantDocuments)

        for rank, document in enumerate(retrievedDocuments, start=1):
            if document in relevantDocumentSet:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def meanReciprocalRank(
        retrievedRankings: Sequence[Sequence[str]],
        relevantDocumentsByCase: Sequence[Collection[str]],
    ) -> float:
        """Calcola la media dei reciprocal rank di tutti i casi sperimentali."""
        if not retrievedRankings:
            raise ValueError("È richiesta almeno una graduatoria di documenti.")

        if len(retrievedRankings) != len(relevantDocumentsByCase):
            raise ValueError("Il numero di graduatorie deve coincidere con il numero dei casi.")

        reciprocalRankSum = 0.0
        for retrievedDocuments, relevantDocuments in zip(
            retrievedRankings,
            relevantDocumentsByCase,
            strict=True,
        ):
            reciprocalRankSum += RetrievalMetricsCalculator.reciprocalRank(
                retrievedDocuments,
                relevantDocuments,
            )

        return reciprocalRankSum / len(retrievedRankings)

    @staticmethod
    def validateK(k: int) -> None:
        if k < 1:
            raise ValueError("Il valore di k deve essere maggiore di zero.")

    @staticmethod
    def toDocumentSet(documents: Collection[str]) -> set[str]:
        documentSet = set(documents)
        if not documentSet:
            raise ValueError("È richiesto almeno un documento rilevante.")
        return documentSet
