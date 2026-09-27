from pydantic import Field, field_validator

from app.models.base import StrictModel
from app.models.observability import TokenUsage


class LLMGenerationResult(StrictModel):
    text: str = Field(min_length=1, description="Testo prodotto dal modello generativo")
    token_usage: TokenUsage | None = Field(
        default=None,
        description="Utilizzo dei token comunicato dal provider, se disponibile",
    )

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value
