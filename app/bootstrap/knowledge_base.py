from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.config import ApplicationConfiguration
from app.indexing import EmbeddingModel, FaissVectorIndex, SentenceTransformerEmbeddingModel
from app.ingestion import LocalDocumentLoader, RegexSensitiveDataMasker, SectionAwareChunker
from app.ingestion.masking import SensitiveDataMasker
from app.models import Document, DocumentChunk

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class IndexingReport:
    document_count: int
    chunk_count: int
    vector_dimension: int
    elapsed_time_ms: float
    output_path: Path


class KnowledgeBaseProcessor:
    """Applica caricamento, mascheramento e chunking alla Knowledge Base."""

    def __init__(
        self,
        configuration: ApplicationConfiguration,
        masker: SensitiveDataMasker | None = None,
    ) -> None:
        self._configuration = configuration
        self._masker = masker or RegexSensitiveDataMasker()

    def load_documents(self) -> list[Document]:
        loader = LocalDocumentLoader(
            self._configuration.ingestion.knowledge_base_path,
            recursive=self._configuration.ingestion.recursive,
        )
        return loader.load()

    def prepare_chunks(self, documents: list[Document] | None = None) -> list[DocumentChunk]:
        source_documents = documents if documents is not None else self.load_documents()
        chunker = SectionAwareChunker(
            chunk_size=self._configuration.chunking.chunk_size_characters,
            chunk_overlap=self._configuration.chunking.chunk_overlap_characters,
        )

        chunks: list[DocumentChunk] = []
        for document in source_documents:
            prepared_document = self._prepare_document(document)
            chunks.extend(chunker.chunk(prepared_document))
        return chunks

    def _prepare_document(self, document: Document) -> Document:
        if not self._configuration.masking.enabled:
            return document

        masked_text = self._masker.mask(document.text)
        return document.model_copy(update={"text": masked_text})


class KnowledgeBaseIndexer:
    """Costruisce e persiste l'indice FAISS usato dal Dense Retriever."""

    def __init__(
        self,
        configuration: ApplicationConfiguration,
        embedding_model: EmbeddingModel | None = None,
        processor: KnowledgeBaseProcessor | None = None,
        clock: Clock = perf_counter,
    ) -> None:
        self._configuration = configuration
        self._embedding_model = embedding_model or self._create_embedding_model()
        self._processor = processor or KnowledgeBaseProcessor(configuration)
        self._clock = clock

    def build(self) -> IndexingReport:
        start_time = self._clock()
        documents = self._processor.load_documents()
        chunks = self._processor.prepare_chunks(documents)
        self._validate_content(documents, chunks)

        vectors = self._embedding_model.encode([chunk.text for chunk in chunks])
        if not vectors or not vectors[0]:
            raise ValueError("Il modello di embedding non ha prodotto vettori indicizzabili.")

        vector_index = FaissVectorIndex(dimension=len(vectors[0]))
        vector_index.add(chunks, vectors)
        vector_index.save(self._configuration.vector_store.path)

        return IndexingReport(
            document_count=len(documents),
            chunk_count=len(chunks),
            vector_dimension=vector_index.dimension,
            elapsed_time_ms=(self._clock() - start_time) * 1000,
            output_path=self._configuration.vector_store.path,
        )

    def _create_embedding_model(self) -> EmbeddingModel:
        embedding_configuration = self._configuration.embeddings
        return SentenceTransformerEmbeddingModel(
            model_name=embedding_configuration.model,
            batch_size=embedding_configuration.batch_size,
            normalize_embeddings=embedding_configuration.normalize,
        )

    @staticmethod
    def _validate_content(
        documents: list[Document],
        chunks: list[DocumentChunk],
    ) -> None:
        if not documents:
            raise ValueError("La Knowledge Base non contiene documenti supportati.")
        if not chunks:
            raise ValueError("La Knowledge Base non ha prodotto chunk indicizzabili.")
