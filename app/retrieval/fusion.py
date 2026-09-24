from collections.abc import Sequence
from dataclasses import dataclass, field
from app.models import DocumentChunk, RetrievalContribution, RetrievalResult

@dataclass
class _FusedCandidate:
    chunk: DocumentChunk
    first_seen_order: int
    fused_score: float = 0.0
    contributions: list[RetrievalContribution] = field(default_factory=list)

    def add(self, result: RetrievalResult, rank_constant: int) -> None:
        self.fused_score += 1.0 / (rank_constant + result.rank)
        self.contributions.append(
            RetrievalContribution(
                retriever=result.retriever,
                rank=result.rank,
                score=result.score,
            )
        )

class ReciprocalRankFusion:
    """Combina più graduatorie usando la posizione dei risultati."""

    DEFAULT_RANK_CONSTANT = 60
    RETRIEVER_NAME = "rrf"

    def __init__(self, rank_constant: int = DEFAULT_RANK_CONSTANT) -> None:
        if rank_constant <= 0:
            raise ValueError("La costante RRF deve essere maggiore di zero.")
        self._rank_constant = rank_constant

    def fuse(
        self,
        rankings: Sequence[Sequence[RetrievalResult]],
    ) -> list[RetrievalResult]:
        candidates = self._collect_candidates(rankings)
        ordered_candidates = sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.fused_score, candidate.first_seen_order),
        )
        return self._map_results(ordered_candidates)

    def _collect_candidates(
        self,
        rankings: Sequence[Sequence[RetrievalResult]],
    ) -> dict[str, _FusedCandidate]:
        candidates: dict[str, _FusedCandidate] = {}
        next_first_seen_order = 0

        for ranking in rankings:
            chunk_ids_in_ranking: set[str] = set()

            for result in ranking:
                chunk_id = result.chunk.id
                if chunk_id in chunk_ids_in_ranking:
                    raise ValueError(
                        f"Il chunk '{chunk_id}' risulta duplicato"
                    )
                chunk_ids_in_ranking.add(chunk_id)

                candidate = candidates.get(chunk_id)
                if candidate is None:
                    candidate = _FusedCandidate(
                        chunk=result.chunk,
                        first_seen_order=next_first_seen_order,
                    )
                    candidates[chunk_id] = candidate
                    next_first_seen_order += 1
                elif candidate.chunk != result.chunk:
                    raise ValueError(
                        f"Il chunk '{chunk_id}' ha contenuti diversi tra le graduatorie"
                    )

                candidate.add(result, self._rank_constant)

        return candidates

    def _map_results(
        self,
        candidates: Sequence[_FusedCandidate],
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []

        for rank, candidate in enumerate(candidates, start=1):
            results.append(
                RetrievalResult(
                    chunk=candidate.chunk,
                    rank=rank,
                    score=None,
                    retriever=self.RETRIEVER_NAME,
                    fused_score=candidate.fused_score,
                    contributions=tuple(candidate.contributions),
                )
            )

        return results