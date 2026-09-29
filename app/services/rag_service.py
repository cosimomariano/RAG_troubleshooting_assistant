from collections.abc import Callable
from time import perf_counter

from app.generation import (
    CitationFormatter,
    LLMClient,
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
from app.validation import RetrievalValidator

Clock = Callable[[], float]


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        promptBuilder: PromptBuilder,
        llmClient: LLMClient,
        topK: int,
        clock: Clock = perf_counter,
    ) -> None:
        self.retriever = retriever
        self.promptBuilder = promptBuilder
        self.llmClient = llmClient
        self.topK = RetrievalValidator.validateTopK(topK)
        self.clock = clock

    def troubleshoot(
        self,
        question: str,
        incidentContext: str | None = None,
    ) -> RAGResponse:
        totalStartTime = self.clock()

        retrievalQuery = self.buildRetrievalQuery(question, incidentContext)
        retrievalExecution = self.retrieveDocuments(retrievalQuery)
        retrievedDocuments = list(retrievalExecution.results)

        promptStartTime = self.clock()
        prompt = self.promptBuilder.build(
            question=question,
            documents=retrievedDocuments,
            incidentContext=incidentContext,
        )
        promptBuildLatencyMs = self.calculateElapsedTimeMs(promptStartTime)

        generationStartTime = self.clock()
        generationResult = self.generateAnswer(prompt)
        generationLatencyMs = self.calculateElapsedTimeMs(generationStartTime)

        sources = self.buildSources(retrievedDocuments)
        totalLatencyMs = self.calculateElapsedTimeMs(totalStartTime)
        operationalMetrics = OperationalMetrics(
            retrievalLatencyMs=retrievalExecution.retrievalLatencyMs,
            rerankingLatencyMs=retrievalExecution.rerankingLatencyMs,
            promptBuildLatencyMs=promptBuildLatencyMs,
            generationLatencyMs=generationLatencyMs,
            totalLatencyMs=totalLatencyMs,
            tokenUsage=generationResult.tokenUsage,
        )

        return RAGResponse(
            answer=generationResult.text,
            sources=sources,
            latencyMs=totalLatencyMs,
            operationalMetrics=operationalMetrics,
        )

    def retrieveDocuments(self, retrievalQuery: str) -> RetrievalExecution:
        if isinstance(self.retriever, MeasuredRetriever):
            return self.retriever.retrieveWithMetrics(retrievalQuery, self.topK)

        retrievalStartTime = self.clock()
        results = self.retriever.retrieve(retrievalQuery, self.topK)
        retrievalLatencyMs = self.calculateElapsedTimeMs(retrievalStartTime)
        return RetrievalExecution(
            results=tuple(results),
            retrievalLatencyMs=retrievalLatencyMs,
        )

    def generateAnswer(self, prompt: str) -> LLMGenerationResult:
        generationResult = self.llmClient.generateWithMetrics(prompt)

        if not generationResult.text:
            raise ValueError("Il client LLM ha restituito una risposta vuota.")
        return generationResult

    @staticmethod
    def buildRetrievalQuery(question: str, incidentContext: str | None) -> str:
        normalizedQuestion = question.strip()
        if not normalizedQuestion:
            raise ValueError("La domanda non può essere vuota.")

        queryParts = [normalizedQuestion]
        if incidentContext is not None:
            normalizedContext = incidentContext.strip()
            if normalizedContext:
                queryParts.append(normalizedContext)
        return "\n".join(queryParts)

    @staticmethod
    def buildSources(documents: list[RetrievalResult]) -> list[SourceReference]:
        sources: list[SourceReference] = []
        for citationPosition, retrievalResult in enumerate(documents, start=1):
            sources.append(RAGService.buildSource(citationPosition, retrievalResult))
        return sources

    @staticmethod
    def buildSource(
        citationPosition: int,
        retrievalResult: RetrievalResult,
    ) -> SourceReference:
        chunk = retrievalResult.chunk
        return SourceReference(
            citationId=CitationFormatter.buildIdentifier(citationPosition),
            documentId=chunk.documentId,
            source=chunk.metadata.source,
            chunkId=chunk.id,
            documentType=chunk.metadata.documentType,
            section=chunk.metadata.section,
            service=chunk.metadata.service,
            category=chunk.metadata.category,
            rank=retrievalResult.rank,
            retriever=retrievalResult.retriever,
            score=retrievalResult.score,
            fusedScore=retrievalResult.fusedScore,
            rerankerScore=retrievalResult.rerankerScore,
        )

    def calculateElapsedTimeMs(self, startTime: float) -> float:
        return (self.clock() - startTime) * 1000
