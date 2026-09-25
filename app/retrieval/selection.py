from enum import StrEnum
from app.retrieval.base import Retriever
from app.retrieval.no_retrieval import NoRetrievalRetriever

# Enum con le possibili modalita di recupero (vedi application.yml)
class RetrievalMode(StrEnum):
    LLM_ONLY = "llm_only"
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


# Selettore del modello di recupero
class RetrieverSelector:
    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        hybrid_retriever: Retriever,
    ) -> None:
        self._retrievers: dict[RetrievalMode, Retriever] = {
            RetrievalMode.LLM_ONLY: NoRetrievalRetriever(),
            RetrievalMode.DENSE: dense_retriever,
            RetrievalMode.SPARSE: sparse_retriever,
            RetrievalMode.HYBRID: hybrid_retriever,
        }

    def select(self, mode: RetrievalMode | str) -> Retriever:
        retrieval_mode = self._parse_mode(mode)
        return self._retrievers[retrieval_mode]

    @staticmethod
    def _parse_mode(mode: RetrievalMode | str) -> RetrievalMode:
        if isinstance(mode, RetrievalMode):
            return mode

        normalized_mode = mode.strip().casefold()
        try:
            return RetrievalMode(normalized_mode)
        #Controllo che la modalita usata sia effettivamente supportata
        except ValueError as error:
            supported_modes = ", ".join(item.value for item in RetrievalMode)
            raise ValueError(
                f"Modalità di retrieval non supportata: '{mode}'. "
                f"Valori ammessi: {supported_modes}."
            ) from error