from math import isclose

from pydantic import Field, model_validator

from app.models.base import StrictModel
from app.models.observability import OperationalMetrics


class SourceReference(StrictModel):
    citationId: str = Field(
        pattern=r"^FONTE_[1-9][0-9]*$",
        description="Identificativo fornito al modello e restituito al client",
    )
    documentId: str = Field(
        min_length=1,
        description="Identificativo stabile del documento originale",
    )
    source: str = Field(min_length=1, description="Documento originale della fonte")
    chunkId: str = Field(min_length=1, description="Identificativo stabile del chunk")
    documentType: str = Field(
        min_length=1,
        description="Tipologia del documento originale",
    )
    section: str | None = Field(default=None, description="Sezione del documento")
    service: str | None = Field(default=None, description="Microservizio associato")
    category: str | None = Field(default=None, description="Categoria documentale associata")
    rank: int = Field(ge=1, description="Posizione nella graduatoria finale")
    retriever: str = Field(
        min_length=1,
        description="Retriever che ha prodotto il risultato finale",
    )
    score: float | None = Field(
        default=None,
        description="Punteggio originale del retriever",
    )
    fusedScore: float | None = Field(
        default=None,
        description="Punteggio prodotto dalla fusione delle graduatorie",
    )
    rerankerScore: float | None = Field(
        default=None,
        description="Punteggio prodotto dal secondo stadio di reranking",
    )


class RAGResponse(StrictModel):
    """Risultato interno dell'orchestrazione RAG."""

    answer: str = Field(min_length=1, description="Risposta generata dal sistema")
    sources: list[SourceReference] = Field(description="Fonti usate nella risposta")
    latencyMs: float = Field(ge=0, description="Latenza totale in millisecondi")
    operationalMetrics: OperationalMetrics = Field(
        description="Dettaglio delle metriche operative della richiesta"
    )

    @model_validator(mode="after")
    def validateTotalLatency(self) -> "RAGResponse":
        if not isclose(
            self.latencyMs,
            self.operationalMetrics.totalLatencyMs,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError("La latenza totale non coincide con le metriche operative.")
        return self
