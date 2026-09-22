from pydantic import Field

from app.models.base import StrictModel
from app.models.documents import DocumentChunk


class RetrievalResult(StrictModel):
    chunk: DocumentChunk
    rank: int = Field(ge=1, description="Posizione del risultato nella graduatoria")
    score: float | None = Field(default=None, description="Punteggio assegnato dal retriever")
    retriever: str = Field(
        min_length=1,
        description="Tipo di retriever che ha prodotto il risultato",
    )
