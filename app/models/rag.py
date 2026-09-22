from pydantic import Field

from app.models.base import StrictModel


class SourceReference(StrictModel):
    source: str = Field(min_length=1, description="Documento originale della fonte")
    chunk_id: str = Field(min_length=1, description="Identificativo stabile del chunk")
    section: str | None = Field(default=None, description="Sezione del documento")
    service: str | None = Field(default=None, description="Microservizio associato")


class RAGResponse(StrictModel):
    """Risultato interno dell'orchestrazione RAG."""

    answer: str = Field(min_length=1, description="Risposta generata dal sistema")
    sources: list[SourceReference] = Field(description="Fonti usate nella risposta")
    latency_ms: float = Field(ge=0, description="Latenza totale in millisecondi")
