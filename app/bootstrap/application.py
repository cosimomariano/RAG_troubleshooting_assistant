from fastapi import FastAPI

from app.api import createApp
from app.bootstrap.knowledge_base import KnowledgeBaseProcessor
from app.config import ApplicationConfiguration
from app.generation import LLMClient, OllamaLLMClient, PromptBuilder
from app.indexing import BM25SparseIndex, EmbeddingModel, FaissVectorIndex
from app.reranking import CrossEncoderReranker, Reranker, RerankingRetriever
from app.retrieval import (
    DenseRetriever,
    HybridRetriever,
    NoRetrievalRetriever,
    ReciprocalRankFusion,
    RetrievalMode,
    Retriever,
    SparseRetriever,
)
from app.services import RAGService


class RAGApplicationFactory:
    """Assembla la pipeline online in base alla configurazione selezionata."""

    def __init__(
        self,
        configuration: ApplicationConfiguration,
        embeddingModel: EmbeddingModel | None = None,
        llmClient: LLMClient | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.configuration = configuration
        self.embeddingModel = embeddingModel
        self.llmClient = llmClient
        self.reranker = reranker

    def createApi(
        self,
        retrievalMode: RetrievalMode | None = None,
        rerankerEnabled: bool | None = None,
    ) -> FastAPI:
        # Creazione del servizio RAG in base alle configurazioni fornite
        ragService = self.createRagService(retrievalMode, rerankerEnabled)

        return createApp(ragService)

    def createRagService(
        self,
        retrievalMode: RetrievalMode | None = None,
        rerankerEnabled: bool | None = None,
    ) -> RAGService:

        # Modalita di retrieval selezionata (llm_only, sparse, dense, hybrid)
        selectedMode = retrievalMode or self.configuration.retrieval.mode
        useReranker = (
            self.configuration.reranker.enabled if rerankerEnabled is None else rerankerEnabled
        )

        candidateCount = self.configuration.retrieval.topK

        # Solo se reranker è abilitato lo valido sulla base dei top-K
        if useReranker:
            candidateCount = self.validateRerankingConfiguration(selectedMode)

        # Creazione del retriever centralizzata
        baseRetriever = self.createBaseRetriever(selectedMode, candidateCount)
        retriever = baseRetriever

        # Se il reranking è abilitato incapsulo il retriever nel wrapper con il candidate count
        if useReranker:
            retriever = self.createRerankingRetriever(
                baseRetriever,
                candidateCount,
            )

        # Istanzio il servizio
        return RAGService(
            retriever=retriever, 
            promptBuilder=PromptBuilder(), # Istanza del costruttore di prompt
            llmClient=self.getLlmClient(), # Istanza del client LLM
            topK=self.configuration.retrieval.topK, # top-K definiti in pagina di configurazione
        )

    def createBaseRetriever(
        self,
        mode: RetrievalMode,
        candidateCount: int,
    ) -> Retriever:
        if mode is RetrievalMode.LLM_ONLY:
            return NoRetrievalRetriever()
        if mode is RetrievalMode.DENSE:
            return self.createDenseRetriever()
        if mode is RetrievalMode.SPARSE:
            return self.createSparseRetriever()
        if mode is RetrievalMode.HYBRID:
            return self.createHybridRetriever(candidateCount)
        raise ValueError(f"Modalità di retrieval non supportata: {mode}.")

    def createDenseRetriever(self) -> DenseRetriever:
        vectorIndex = FaissVectorIndex.load(self.configuration.vectorStore.path)
        return DenseRetriever(
            embeddingModel=self.getEmbeddingModel(),
            vectorIndex=vectorIndex,
        )

    def createSparseRetriever(self) -> SparseRetriever:
        chunks = KnowledgeBaseProcessor(self.configuration).prepareChunks()
        sparseIndex = BM25SparseIndex(chunks)
        return SparseRetriever(sparseIndex)

    def createHybridRetriever(self, candidateCount: int) -> HybridRetriever:
        return HybridRetriever(
            sparseRetriever=self.createSparseRetriever(),
            denseRetriever=self.createDenseRetriever(),
            rankFusion=ReciprocalRankFusion(),
            sparseTopK=candidateCount,
            denseTopK=candidateCount,
        )

    def createRerankingRetriever(
        self,
        baseRetriever: Retriever,
        candidateCount: int,
    ) -> RerankingRetriever:
        return RerankingRetriever(
            candidateRetriever=baseRetriever,
            reranker=self.getReranker(),
            candidateTopN=candidateCount,
        )

    # Controlli di base per il meccanismo di reranking
    def validateRerankingConfiguration(self, mode: RetrievalMode) -> int:
        if mode is RetrievalMode.LLM_ONLY:
            raise ValueError("Il reranking non può essere applicato al baseline LLM-only.")

        candidateTopN = self.configuration.reranker.candidateTopN
        if candidateTopN < self.configuration.retrieval.topK:
            raise ValueError("Il Top-N del reranker non può essere minore del Top-K finale.")
        return candidateTopN

    def getEmbeddingModel(self) -> EmbeddingModel:
        if self.embeddingModel is None:
            embeddingConfiguration = self.configuration.embeddings
            from app.indexing import SentenceTransformerEmbeddingModel

            self.embeddingModel = SentenceTransformerEmbeddingModel(
                modelName=embeddingConfiguration.model,
                batchSize=embeddingConfiguration.batchSize,
                normalizeEmbeddings=embeddingConfiguration.normalize,
                inputPrefix=embeddingConfiguration.queryPrefix,
            )
        return self.embeddingModel

    def getLlmClient(self) -> LLMClient:
        if self.llmClient is None:
            # Puntamento alla sezione llm della configurazione per il prelievo degli iperparametri del foundation model
            llmConfiguration = self.configuration.llm
            
            self.llmClient = OllamaLLMClient(
                baseUrl=llmConfiguration.baseUrl,
                model=llmConfiguration.model,
                timeoutSeconds=llmConfiguration.timeoutSeconds,
            )
        return self.llmClient

    def getReranker(self) -> Reranker:
        if self.reranker is None:
            rerankerConfiguration = self.configuration.reranker

            # Istanzio il CrossEncoderReranker sulla base del modello e batch size definiti in configuration
            self.reranker = CrossEncoderReranker(
                modelName=rerankerConfiguration.model,
                batchSize=rerankerConfiguration.batchSize,
            )
        return self.reranker
