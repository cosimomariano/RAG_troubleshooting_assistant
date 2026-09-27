from fastapi import FastAPI

from app.api import create_app
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
        embedding_model: EmbeddingModel | None = None,
        llm_client: LLMClient | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._configuration = configuration
        self._embedding_model = embedding_model
        self._llm_client = llm_client
        self._reranker = reranker

    def create_api(
        self,
        retrieval_mode: RetrievalMode | None = None,
        reranker_enabled: bool | None = None,
    ) -> FastAPI:
        rag_service = self.create_rag_service(retrieval_mode, reranker_enabled)
        return create_app(rag_service)

    def create_rag_service(
        self,
        retrieval_mode: RetrievalMode | None = None,
        reranker_enabled: bool | None = None,
    ) -> RAGService:
        selected_mode = retrieval_mode or self._configuration.retrieval.mode
        use_reranker = (
            self._configuration.reranker.enabled
            if reranker_enabled is None
            else reranker_enabled
        )

        retriever = self._create_retriever(selected_mode, use_reranker)
        retriever = self._apply_reranker(retriever, selected_mode, use_reranker)
        return RAGService(
            retriever=retriever,
            prompt_builder=PromptBuilder(),
            llm_client=self._get_llm_client(),
            top_k=self._configuration.retrieval.top_k,
        )

    def _create_retriever(
        self,
        mode: RetrievalMode,
        reranker_enabled: bool,
    ) -> Retriever:
        if mode is RetrievalMode.LLM_ONLY:
            return NoRetrievalRetriever()
        if mode is RetrievalMode.DENSE:
            return self._create_dense_retriever()
        if mode is RetrievalMode.SPARSE:
            return self._create_sparse_retriever()
        if mode is RetrievalMode.HYBRID:
            return self._create_hybrid_retriever(reranker_enabled)
        raise ValueError(f"Modalità di retrieval non supportata: {mode}.")

    def _create_dense_retriever(self) -> DenseRetriever:
        vector_index = FaissVectorIndex.load(self._configuration.vector_store.path)
        return DenseRetriever(
            embedding_model=self._get_embedding_model(),
            vector_index=vector_index,
        )

    def _create_sparse_retriever(self) -> SparseRetriever:
        chunks = KnowledgeBaseProcessor(self._configuration).prepare_chunks()
        sparse_index = BM25SparseIndex(chunks)
        return SparseRetriever(sparse_index)

    def _create_hybrid_retriever(self, reranker_enabled: bool) -> HybridRetriever:
        candidate_count = self._configuration.retrieval.top_k
        if reranker_enabled:
            candidate_count = max(
                candidate_count,
                self._configuration.reranker.candidate_top_n,
            )

        return HybridRetriever(
            sparse_retriever=self._create_sparse_retriever(),
            dense_retriever=self._create_dense_retriever(),
            rank_fusion=ReciprocalRankFusion(),
            sparse_top_k=candidate_count,
            dense_top_k=candidate_count,
        )

    def _apply_reranker(
        self,
        retriever: Retriever,
        mode: RetrievalMode,
        reranker_enabled: bool,
    ) -> Retriever:
        if not reranker_enabled:
            return retriever
        if mode is RetrievalMode.LLM_ONLY:
            raise ValueError("Il reranking non può essere applicato al baseline LLM-only.")

        candidate_top_n = self._configuration.reranker.candidate_top_n
        if candidate_top_n < self._configuration.retrieval.top_k:
            raise ValueError("Il Top-N del reranker non può essere minore del Top-K finale.")

        return RerankingRetriever(
            candidate_retriever=retriever,
            reranker=self._get_reranker(),
            candidate_top_n=candidate_top_n,
        )

    def _get_embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            embedding_configuration = self._configuration.embeddings
            from app.indexing import SentenceTransformerEmbeddingModel

            self._embedding_model = SentenceTransformerEmbeddingModel(
                model_name=embedding_configuration.model,
                batch_size=embedding_configuration.batch_size,
                normalize_embeddings=embedding_configuration.normalize,
            )
        return self._embedding_model

    def _get_llm_client(self) -> LLMClient:
        if self._llm_client is None:
            llm_configuration = self._configuration.llm
            self._llm_client = OllamaLLMClient(
                base_url=llm_configuration.base_url,
                model=llm_configuration.model,
                timeout_seconds=llm_configuration.timeout_seconds,
            )
        return self._llm_client

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            reranker_configuration = self._configuration.reranker
            self._reranker = CrossEncoderReranker(
                model_name=reranker_configuration.model,
                batch_size=reranker_configuration.batch_size,
            )
        return self._reranker
