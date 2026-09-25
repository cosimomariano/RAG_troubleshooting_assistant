from pydantic import Field

from app.models.base import StrictModel
from app.models.documents import DocumentChunk


class RetrievalContribution(StrictModel):
    retriever: str = Field(
        min_length=1,
        description="Retriever che ha incluso il chunk",
    )
    rank: int = Field(ge=1, description="Posizione del chunk nella ranking originale")
    score: float | None = Field(
        default=None,
        description="Punteggio originale del retriever",
    )


class RetrievalResult(StrictModel):
    chunk: DocumentChunk
    rank: int = Field(ge=1, description="Posizione del risultato nella graduatoria")
    score: float | None = Field(default=None, description="Punteggio assegnato dal retriever")
    retriever: str = Field(
        min_length=1,
        description="Tipo di retriever che ha prodotto il risultato",
    )
    fused_score: float | None = Field(
        default=None,
        description="Punteggio prodotto dalla fusione di più graduatorie",
    )
    reranker_score: float | None = Field(
        default=None,
        description="Punteggio assegnato dal modello di reranking",
    )
    contributions: tuple[RetrievalContribution, ...] = Field(
        default=(),
        description="Rank e punteggi originali che hanno contribuito alla fusione",
    )
