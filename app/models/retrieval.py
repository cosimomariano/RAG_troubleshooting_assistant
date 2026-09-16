from pydantic import BaseModel, ConfigDict, Field
from app.models.documents import DocumentChunk

"MOdello per il retriver"
class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid") # forbid per impedire la silent injection di campi non previsti nei modelli per evitare divergenze

    chunk: DocumentChunk
    rank: int = Field(ge=1, description="Posizione del risultato nella classifica(posizionamento/rank)")
    score: float | None = Field(default=None, description="Punteggio assegnato dal retriever")
    retriever: str = Field(min_length=1, description="Tipo di retriever che ha prodotto il risultato")