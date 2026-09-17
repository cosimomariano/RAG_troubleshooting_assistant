from pydantic import BaseModel, ConfigDict, Field

class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid") # forbid per impedire la silent injection di campi non previsti nei modelli per evitare divergenze
    source: str = Field(min_length=1, description="Percorso/nome del file")
    document_type: str = Field(min_length=1, description="Identificativo della tipologia associata alla fonte documentale")
    service: str | None = Field(default=None, description="Microservizio associato alla fonte")
    section: str | None = Field(default=None, description="Identificativo della sorgente documentale")
    category: str | None = Field(default=None, description="Identificativo della categoria documentale")

class Document(BaseModel):
    model_config = ConfigDict(extra="forbid") # forbid per impedire la silent injection di campi non previsti nei modelli per evitare divergenze

    id: str = Field(min_length=1, description= "Identificativo univoco associato al documento.")
    text: str = Field(description ="Contenuto del documento in formato testuale")
    metadata: SourceMetadata

"Modello per il chunking"
class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid") # forbid per impedire la silent injection di campi non previsti nei modelli per evitare divergenze

    id: str = Field(min_length=1, description="Identificativo univoco del chunk di riferimento")
    document_id: str = Field(min_length=1, description="Identificativo del documento dal quale viene staccato il chunk")
    text: str = Field(min_length=1, description="Contenuto testuale del chunk")
    metadata: SourceMetadata