from time import perf_counter
from app.generation import LLMClient, PromptBuilder
from app.models import RAGResponse, RetrievalResult, SourceReference
from app.retrieval import Retriever

# Servizio di rag per l'orchestrazione del flusso dense
class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_client: LLMClient,
        top_k: int,
    ) -> None:
        if top_k <= 0:
            raise ValueError("Il valore Top-K deve essere maggiore di zero.")

        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_client = llm_client
        self._top_k = top_k

    def troubleshoot(
        self,
        question: str,
        incident_context: str | None = None,
    ) -> RAGResponse:
        """Esegue il flusso RAG e restituisce risposta, fonti e latenza."""

        started_at = perf_counter()

        retrieval_query = self._build_retrieval_query(question, incident_context)
        retrieved_documents = self._retriever.retrieve(retrieval_query, self._top_k)

        prompt = self._prompt_builder.build(
            question=question,
            documents=retrieved_documents,
            incident_context=incident_context,
        )
        answer = self._llm_client.generate(prompt).strip()
        if not answer:
            raise ValueError("Il client LLM ha restituito una risposta vuota.")

        elapsed_ms = (perf_counter() - started_at) * 1000
        return RAGResponse(
            answer=answer,
            sources=self._build_sources(retrieved_documents),
            latency_ms=elapsed_ms,
        )

    @staticmethod
    def _build_retrieval_query(question: str, incident_context: str | None) -> str:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("La domanda non può essere vuota.")

        query_parts = [normalized_question]
        if incident_context and incident_context.strip():
            query_parts.append(incident_context.strip())

        return "\n".join(query_parts)

    @staticmethod
    def _build_sources(documents: list[RetrievalResult]) -> list[SourceReference]:
        return [
            SourceReference(
                source=result.chunk.metadata.source,
                chunk_id=result.chunk.id,
                section=result.chunk.metadata.section,
                service=result.chunk.metadata.service,
            )
            for result in documents
        ]