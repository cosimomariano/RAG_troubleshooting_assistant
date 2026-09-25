from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol
from app.models import RetrievalResult

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
class _ScoredCandidate:
    result: RetrievalResult
    score: float
    original_order: int

class CrossEncoderReranker:
    """Valuta congiuntamente la query e il testo di ogni chunk candidato."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        backend: CrossEncoderBackend | None = None,
    ) -> None:
        self._model_name = self._validate_model_name(model_name)
        self._batch_size = self._validate_batch_size(batch_size)
        self._backend = self._resolve_backend(backend)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        normalized_query = self._normalize_query(query)
        final_top_k = self._validate_top_k(top_k)
        candidate_list = list(candidates)

        if not candidate_list:
            return []

        text_pairs = self._build_text_pairs(normalized_query, candidate_list)
        scores = self._predict_scores(text_pairs)
        scored_candidates = self._join_candidates_and_scores(candidate_list, scores)
        ordered_candidates = self._sort_candidates(scored_candidates)
        return self._build_results(ordered_candidates, final_top_k)

    def _predict_scores(self, text_pairs: list[TextPair]) -> list[float]:
        raw_scores = self._backend.predict(
            text_pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = self._convert_scores(raw_scores)
        if len(scores) != len(text_pairs):
            raise ValueError(
                "Il numero di punteggi del Cross-Encoder non coincide con i candidati."
            )
        return scores

    @staticmethod
    def _build_text_pairs(
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[TextPair]:
        return [(query, candidate.chunk.text) for candidate in candidates]

    @staticmethod
    def _convert_scores(raw_scores: Sequence[float]) -> list[float]:
        scores: list[float] = []
        for raw_score in raw_scores:
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Il Cross-Encoder deve restituire un punteggio numerico per candidato."
                ) from error

            if not isfinite(score):
                raise ValueError("Il Cross-Encoder ha restituito un punteggio non finito.")
            scores.append(score)
        return scores

    @staticmethod
    def _join_candidates_and_scores(
        candidates: Sequence[RetrievalResult],
        scores: Sequence[float],
    ) -> list[_ScoredCandidate]:
        return [
            _ScoredCandidate(
                result=candidate,
                score=scores[index],
                original_order=index,
            )
            for index, candidate in enumerate(candidates)
        ]

    @staticmethod
    def _sort_candidates(
        candidates: Sequence[_ScoredCandidate],
    ) -> list[_ScoredCandidate]:
        return sorted(
            candidates,
            key=lambda candidate: (-candidate.score, candidate.original_order),
        )

    @staticmethod
    def _build_results(
        candidates: Sequence[_ScoredCandidate],
        top_k: int,
    ) -> list[RetrievalResult]:
        return [
            candidate.result.model_copy(
                update={
                    "rank": rank,
                    "reranker_score": candidate.score,
                }
            )
            for rank, candidate in enumerate(candidates[:top_k], start=1)
        ]

    def _resolve_backend(
        self,
        backend: CrossEncoderBackend | None,
    ) -> CrossEncoderBackend:
        if backend is not None:
            return backend
        return self._load_backend(self._model_name)

    @staticmethod
    def _load_backend(model_name: str) -> CrossEncoderBackend:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError("Impossibile caricare Sentence Transformers.") from error

        return CrossEncoder(model_name)

    @staticmethod
    def _validate_model_name(model_name: str) -> str:
        normalized_model_name = model_name.strip()
        if not normalized_model_name:
            raise ValueError("Il nome del modello Cross-Encoder non può essere vuoto.")
        return normalized_model_name

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("La dimensione del batch deve essere maggiore di zero.")
        return batch_size

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if top_k <= 0:
            raise ValueError("Il valore Top-K deve essere maggiore di zero.")
        return top_k

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("La query non può essere vuota.")
        return normalized_query