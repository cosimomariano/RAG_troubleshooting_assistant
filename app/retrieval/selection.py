from app.retrieval.base import Retriever
from app.retrieval.mode import RetrievalMode
from app.retrieval.no_retrieval import NoRetrievalRetriever


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
        except ValueError as error:
            supported_modes = ", ".join(item.value for item in RetrievalMode)
            raise ValueError(
                f"Modalità di retrieval non supportata: '{mode}'. "
                f"Valori ammessi: {supported_modes}."
            ) from error