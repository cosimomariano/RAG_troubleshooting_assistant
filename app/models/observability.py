from pydantic import Field, model_validator

from app.models.base import StrictModel


class TokenUsage(StrictModel):
    input_tokens: int = Field(
        ge=0,
        description="Numero di token elaborati nel prompt",
    )
    output_tokens: int = Field(
        ge=0,
        description="Numero di token generati nella risposta",
    )
    total_tokens: int = Field(
        ge=0,
        description="Somma dei token di input e di output",
    )

    @model_validator(mode="after")
    def validate_total_tokens(self) -> "TokenUsage":
        expected_total = self.input_tokens + self.output_tokens
        if self.total_tokens != expected_total:
            raise ValueError("Il totale dei token non coincide con input e output.")
        return self


class OperationalMetrics(StrictModel):
    retrieval_latency_ms: float = Field(
        ge=0,
        description="Latenza del recupero documentale in millisecondi",
    )
    reranking_latency_ms: float = Field(
        ge=0,
        description="Latenza del secondo stadio di reranking in millisecondi",
    )
    prompt_build_latency_ms: float = Field(
        ge=0,
        description="Latenza della costruzione del prompt in millisecondi",
    )
    generation_latency_ms: float = Field(
        ge=0,
        description="Latenza della generazione LLM in millisecondi",
    )
    total_latency_ms: float = Field(
        ge=0,
        description="Latenza complessiva della richiesta in millisecondi",
    )
    token_usage: TokenUsage | None = Field(
        default=None,
        description="Utilizzo dei token comunicato dal provider LLM, se disponibile",
    )
