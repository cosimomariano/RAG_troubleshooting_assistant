from collections.abc import Callable
from time import perf_counter

from app.generation import (
    CitationFormatter,
    LLMClient,
    MeasuredLLMClient,
    PromptBuilder,
)
from app.models import (
    LLMGenerationResult,
    OperationalMetrics,
    RAGResponse,
    RetrievalResult,
    SourceReference,
)
from app.retrieval import MeasuredRetriever, RetrievalExecution, Retriever

Clock = Callable[[], float]


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        top_k: int,
        clock: Clock = perf_counter,
    ) -> None:
        self._validate_top_k(top_k)
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_client = llm_client
        self._top_k = top_k
        self._clock = clock

    def troubleshoot(
        self,
        question: str,
        incident_context: str | None = None,
    ) -> RAGResponse:
        total_start_time = self._clock()

        retrieval_query = self._build_retrieval_query(question, incident_context)
        retrieval_execution = self._retrieve_documents(retrieval_query)

        prompt_start_time = self._clock()
        prompt = self._build_prompt(
            question,
            incident_context,
            list(retrieval_execution.results),
        )
        prompt_build_latency_ms = self._calculate_elapsed_time_ms(prompt_start_time)

        generation_start_time = self._clock()
        generation_result = self._generate_answer(prompt)
        generation_latency_ms = self._calculate_elapsed_time_ms(generation_start_time)

        sources = self._build_sources(list(retrieval_execution.results))
        total_latency_ms = self._calculate_elapsed_time_ms(total_start_time)
        operational_metrics = OperationalMetrics(
            retrieval_latency_ms=retrieval_execution.retrieval_latency_ms,
            reranking_latency_ms=retrieval_execution.reranking_latency_ms,
            prompt_build_latency_ms=prompt_build_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            token_usage=generation_result.token_usage,
        )

        return RAGResponse(
            answer=generation_result.text,
            sources=sources,
            latency_ms=total_latency_ms,
            operational_metrics=operational_metrics,
        )

    def _retrieve_documents(self, retrieval_query: str) -> RetrievalExecution:
        if isinstance(self._retriever, MeasuredRetriever):
            return self._retriever.retrieve_with_metrics(retrieval_query, self._top_k)

        retrieval_start_time = self._clock()
        results = self._retriever.retrieve(retrieval_query, self._top_k)
        retrieval_latency_ms = self._calculate_elapsed_time_ms(retrieval_start_time)
        return RetrievalExecution(
            results=tuple(results),
            retrieval_latency_ms=retrieval_latency_ms,
        )

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

    def _generate_answer(self, prompt: str) -> LLMGenerationResult:
        if isinstance(self._llm_client, MeasuredLLMClient):
            generation_result = self._llm_client.generate_with_metrics(prompt)
        else:
            generated_text = self._llm_client.generate(prompt).strip()
            if not generated_text:
                raise ValueError("Il client LLM ha restituito una risposta vuota.")
            generation_result = LLMGenerationResult(
                text=generated_text,
            )

        if not generation_result.text:
            raise ValueError("Il client LLM ha restituito una risposta vuota.")
        return generation_result

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
        for citation_position, retrieval_result in enumerate(documents, start=1):
            sources.append(
                RAGService._build_source(citation_position, retrieval_result)
            )
        return sources

    @staticmethod
    def _build_source(
        citation_position: int,
        retrieval_result: RetrievalResult,
    ) -> SourceReference:
        chunk = retrieval_result.chunk
        return SourceReference(
            citation_id=CitationFormatter.build_identifier(citation_position),
            document_id=chunk.document_id,
            source=chunk.metadata.source,
            chunk_id=chunk.id,
            document_type=chunk.metadata.document_type,
            section=chunk.metadata.section,
            service=chunk.metadata.service,
            category=chunk.metadata.category,
            rank=retrieval_result.rank,
            retriever=retrieval_result.retriever,
            score=retrieval_result.score,
            fused_score=retrieval_result.fused_score,
            reranker_score=retrieval_result.reranker_score,
        )

    def _calculate_elapsed_time_ms(self, start_time: float) -> float:
        return (self._clock() - start_time) * 1000

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("Il valore Top-K deve essere maggiore di zero.")
