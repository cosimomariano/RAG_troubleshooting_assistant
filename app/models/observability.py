from pydantic import Field, model_validator

from app.models.base import StrictModel


class TokenUsage(StrictModel):
    inputTokens: int = Field(
        ge=0,
        description="Numero di token elaborati nel prompt",
    )
    outputTokens: int = Field(
        ge=0,
        description="Numero di token generati nella risposta",
    )
    totalTokens: int = Field(
        ge=0,
        description="Somma dei token di input e di output",
    )

    @model_validator(mode="after")
    def validateTotalTokens(self) -> "TokenUsage":
        expectedTotal = self.inputTokens + self.outputTokens
        if self.totalTokens != expectedTotal:
            raise ValueError("Il totale dei token non coincide con input e output.")
        return self


class OperationalMetrics(StrictModel):
    retrievalLatencyMs: float = Field(
        ge=0,
        description="Latenza del recupero documentale in millisecondi",
    )
    rerankingLatencyMs: float = Field(
        ge=0,
        description="Latenza del secondo stadio di reranking in millisecondi",
    )
    promptBuildLatencyMs: float = Field(
        ge=0,
        description="Latenza della costruzione del prompt in millisecondi",
    )
    generationLatencyMs: float = Field(
        ge=0,
        description="Latenza della generazione LLM in millisecondi",
    )
    totalLatencyMs: float = Field(
        ge=0,
        description="Latenza complessiva della richiesta in millisecondi",
    )
    tokenUsage: TokenUsage | None = Field(
        default=None,
        description="Utilizzo dei token comunicato dal provider LLM, se disponibile",
    )
