from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.models import DocumentChunk, RetrievalContribution, RetrievalResult


@runtime_checkable
class RankFusion(Protocol):
    # Contratto per la combinazione di piu graduatore (dense + sparse)

    def fuse(
        self,
        rankings: Sequence[Sequence[RetrievalResult]],
    ) -> list[RetrievalResult]: ...


@dataclass
class FusedCandidate:
    chunk: DocumentChunk
    firstSeenOrder: int
    fusedScore: float = 0.0
    contributions: list[RetrievalContribution] = field(default_factory=list)

    def add(self, result: RetrievalResult, rankConstant: int) -> None:
        self.fusedScore += 1.0 / (rankConstant + result.rank)
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

    def __init__(self, rankConstant: int = DEFAULT_RANK_CONSTANT) -> None:
        if rankConstant <= 0:
            raise ValueError("La costante RRF deve essere maggiore di zero.")
        self.rankConstant = rankConstant

    def fuse(
        self,
        rankings: Sequence[Sequence[RetrievalResult]],
    ) -> list[RetrievalResult]:
        # Validazione e collezione dei rank candidati
        candidates = self.collectCandidates(rankings)

        # Ordinamento dei candidati dallo score piu alto al piu basso
        orderedCandidates = sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.fusedScore, candidate.firstSeenOrder),
        )
        return self.mapResults(orderedCandidates)

    def collectCandidates(
        self,
        rankings: Sequence[Sequence[RetrievalResult]],
    ) -> dict[str, FusedCandidate]:
        candidates: dict[str, FusedCandidate] = {}
        nextFirstSeenOrder = 0

        for ranking in rankings:
            chunkIdsInRanking: set[str] = set()

            for result in ranking:
                chunkId = result.chunk.id
                if chunkId in chunkIdsInRanking:
                    raise ValueError(f"Il chunk '{chunkId}' risulta duplicato")
                chunkIdsInRanking.add(chunkId)

                candidate = candidates.get(chunkId)
                if candidate is None:
                    candidate = FusedCandidate(
                        chunk=result.chunk,
                        firstSeenOrder=nextFirstSeenOrder,
                    )
                    candidates[chunkId] = candidate
                    nextFirstSeenOrder += 1
                elif candidate.chunk != result.chunk:
                    raise ValueError(
                        f"Il chunk '{chunkId}' ha contenuti diversi tra le graduatorie"
                    )

                candidate.add(result, self.rankConstant)

        return candidates

    def mapResults(
        self,
        candidates: Sequence[FusedCandidate],
    ) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []

        # Incapsulamento in oggetto RetrievalResult per lavorazioni successive
        for rank, candidate in enumerate(candidates, start=1):
            results.append(
                RetrievalResult(
                    chunk=candidate.chunk,
                    rank=rank,
                    score=None,
                    retriever=self.RETRIEVER_NAME,
                    fusedScore=candidate.fusedScore,
                    contributions=tuple(candidate.contributions),
                )
            )

        return results
