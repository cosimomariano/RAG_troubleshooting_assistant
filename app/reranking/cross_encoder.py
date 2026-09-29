from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from app.models import RetrievalResult
from app.validation import RetrievalValidator

TextPair = tuple[str, str]


class CrossEncoderBackend(Protocol):
    def predict(
        self,
        inputs: list[TextPair],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class ScoredCandidate:
    result: RetrievalResult
    score: float
    originalOrder: int


class CrossEncoderReranker:
    """Valuta congiuntamente la query e il testo di ogni chunk candidato."""

    def __init__(
        self,
        modelName: str,
        *,
        batchSize: int = 32,
        backend: CrossEncoderBackend | None = None,
    ) -> None:
        self.modelName = self.validateModelName(modelName)
        self.batchSize = self.validateBatchSize(batchSize)
        self.backend = self.resolveBackend(backend)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        topK: int,
    ) -> list[RetrievalResult]:
        normalizedQuery = RetrievalValidator.normalizeQuery(query)
        finalTopK = RetrievalValidator.validateTopK(topK)
        candidateList = list(candidates)

        if not candidateList:
            return []

        textPairs = self.buildTextPairs(normalizedQuery, candidateList)
        scores = self.predictScores(textPairs)
        scoredCandidates = self.joinCandidatesAndScores(candidateList, scores)
        orderedCandidates = self.sortCandidates(scoredCandidates)
        return self.buildResults(orderedCandidates, finalTopK)

    def predictScores(self, textPairs: list[TextPair]) -> list[float]:
        rawScores = self.backend.predict(
            textPairs,
            batch_size=self.batchSize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = self.convertScores(rawScores)
        if len(scores) != len(textPairs):
            raise ValueError(
                "Il numero di punteggi del Cross-Encoder non coincide con i candidati."
            )
        return scores

    @staticmethod
    def buildTextPairs(
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[TextPair]:
        return [(query, candidate.chunk.text) for candidate in candidates]

    @staticmethod
    def convertScores(rawScores: Sequence[float]) -> list[float]:
        scores: list[float] = []
        for rawScore in rawScores:
            try:
                score = float(rawScore)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Il Cross-Encoder deve restituire un punteggio numerico per candidato."
                ) from error

            if not isfinite(score):
                raise ValueError("Il Cross-Encoder ha restituito un punteggio non finito.")
            scores.append(score)
        return scores

    @staticmethod
    def joinCandidatesAndScores(
        candidates: Sequence[RetrievalResult],
        scores: Sequence[float],
    ) -> list[ScoredCandidate]:
        return [
            ScoredCandidate(
                result=candidate,
                score=scores[index],
                originalOrder=index,
            )
            for index, candidate in enumerate(candidates)
        ]

    @staticmethod
    def sortCandidates(
        candidates: Sequence[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (-candidate.score, candidate.originalOrder),
        )

    @staticmethod
    def buildResults(
        candidates: Sequence[ScoredCandidate],
        topK: int,
    ) -> list[RetrievalResult]:
        return [
            candidate.result.model_copy(
                update={
                    "rank": rank,
                    "rerankerScore": candidate.score,
                }
            )
            for rank, candidate in enumerate(candidates[:topK], start=1)
        ]

    def resolveBackend(
        self,
        backend: CrossEncoderBackend | None,
    ) -> CrossEncoderBackend:
        if backend is not None:
            return backend
        return self.loadBackend(self.modelName)

    @staticmethod
    def loadBackend(modelName: str) -> CrossEncoderBackend:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError("Impossibile caricare Sentence Transformers.") from error

        return CrossEncoder(modelName)

    @staticmethod
    def validateModelName(modelName: str) -> str:
        normalizedModelName = modelName.strip()
        if not normalizedModelName:
            raise ValueError("Il nome del modello Cross-Encoder non può essere vuoto.")
        return normalizedModelName

    @staticmethod
    def validateBatchSize(batchSize: int) -> int:
        if batchSize <= 0:
            raise ValueError("La dimensione del batch deve essere maggiore di zero.")
        return batchSize
