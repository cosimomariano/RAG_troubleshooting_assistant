from time import perf_counter

from app.generation import LLMClient, PromptBuilder
from app.models import RAGResponse, RetrievalResult, SourceReference
from app.retrieval import Retriever


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        top_k: int,
    ) -> None:
        self._validate_top_k(top_k)
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_client = llm_client
        self._top_k = top_k

    def troubleshoot(
        self,
        question: str,
        incident_context: str | None = None,
    ) -> RAGResponse:
        start_time = perf_counter()

        retrieval_query = self._build_retrieval_query(question, incident_context)
        retrieved_documents = self._retrieve_documents(retrieval_query)
        prompt = self._build_prompt(question, incident_context, retrieved_documents)
        answer = self._generate_answer(prompt)
        elapsed_time_ms = self._calculate_elapsed_time_ms(start_time)

        return RAGResponse(
            answer=answer,
            sources=self._build_sources(retrieved_documents),
            latency_ms=elapsed_time_ms,
        )

    def _retrieve_documents(self, retrieval_query: str) -> list[RetrievalResult]:
        return self._retriever.retrieve(retrieval_query, self._top_k)

    def _build_prompt(
        self,
        question: str,
        incident_context: str | None,
        retrieved_documents: list[RetrievalResult],
    ) -> str:
        return self._prompt_builder.build(
            question=question,
            documents=retrieved_documents,
            incident_context=incident_context,
        )

    def _generate_answer(self, prompt: str) -> str:
        answer = self._llm_client.generate(prompt).strip()
        if not answer:
            raise ValueError("Il client LLM ha restituito una risposta vuota.")
        return answer

    @staticmethod
    def _build_retrieval_query(question: str, incident_context: str | None) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("La domanda non può essere vuota.")

        query_parts = [normalized_question]
        if incident_context is not None:
            normalized_context = incident_context.strip()
            if normalized_context:
                query_parts.append(normalized_context)
        return "\n".join(query_parts)

    @staticmethod
    def _build_sources(documents: list[RetrievalResult]) -> list[SourceReference]:
        sources: list[SourceReference] = []
        for retrieval_result in documents:
            sources.append(RAGService._build_source(retrieval_result))
        return sources

    @staticmethod
    def _build_source(retrieval_result: RetrievalResult) -> SourceReference:
        chunk = retrieval_result.chunk
        return SourceReference(
            source=chunk.metadata.source,
            chunk_id=chunk.id,
            section=chunk.metadata.section,
            service=chunk.metadata.service,
        )

    @staticmethod
    def _calculate_elapsed_time_ms(start_time: float) -> float:
        return (perf_counter() - start_time) * 1000

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("Il valore Top-K deve essere maggiore di zero.")
