from enum import StrEnum


class RetrievalMode(StrEnum):
    """Modalità di recupero selezionabili dalla configurazione."""

    LLM_ONLY = "llm_only"
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
