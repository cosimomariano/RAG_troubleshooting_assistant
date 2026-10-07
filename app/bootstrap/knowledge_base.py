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
    documentCount: int
    chunkCount: int
    vectorDimension: int
    elapsedTimeMs: float
    outputPath: Path


class KnowledgeBaseProcessor:
    """Applica caricamento, mascheramento e chunking alla Knowledge Base."""

    def __init__(
        self,
        configuration: ApplicationConfiguration,
        masker: SensitiveDataMasker | None = None,
    ) -> None:
        self.configuration = configuration
        self.masker = masker or RegexSensitiveDataMasker()

    def loadDocuments(self) -> list[Document]:
        loader = LocalDocumentLoader(
            self.configuration.ingestion.knowledgeBasePath,
            recursive=self.configuration.ingestion.recursive,
        )
        return loader.load()

    def prepareChunks(self, documents: list[Document] | None = None) -> list[DocumentChunk]:
        sourceDocuments = documents if documents is not None else self.loadDocuments()
        chunker = SectionAwareChunker(
            chunkSize=self.configuration.chunking.chunkSizeCharacters,
            chunkOverlap=self.configuration.chunking.chunkOverlapCharacters,
        )

        chunks: list[DocumentChunk] = []
        for document in sourceDocuments:
            preparedDocument = self.prepareDocument(document)
            chunks.extend(chunker.chunk(preparedDocument))
        return chunks

    def prepareDocument(self, document: Document) -> Document:
        if not self.configuration.masking.enabled:
            return document

        maskedText = self.masker.mask(document.text)
        return document.model_copy(update={"text": maskedText})


class KnowledgeBaseIndexer:
    """Costruisce e persiste l'indice FAISS usato dal Dense Retriever."""

    def __init__(
        self,
        configuration: ApplicationConfiguration,
        embeddingModel: EmbeddingModel | None = None,
        processor: KnowledgeBaseProcessor | None = None,
        clock: Clock = perf_counter,
    ) -> None:
        self.configuration = configuration
        self.embeddingModel = embeddingModel or self.createEmbeddingModel()
        self.processor = processor or KnowledgeBaseProcessor(configuration)
        self.clock = clock

    def build(self) -> IndexingReport:
        startTime = self.clock()
        # Carico i documenti dalla cartella della KB
        documents = self.processor.loadDocuments()
        # Divido in chunk la documentazione usando section-aware chunking
        # effettuando prima il masking dei dati sensibili con l'utilizzo di regex deterministiche
        chunks = self.processor.prepareChunks(documents)
        # Validazione del contenuto
        self.validateContent(documents, chunks)

        # Codifica dei chunk utilizzando il modello di embedding fornito in configurazione
        vectors = self.embeddingModel.encode([chunk.text for chunk in chunks])
        if not vectors or not vectors[0]:
            raise ValueError("Il modello di embedding non ha prodotto vettori indicizzabili.")

        # Istanzio il FaissVectorIndex che utilizza la libreria Faiss
        vectorIndex = FaissVectorIndex(dimension=len(vectors[0]))

        # Aggiungo vettori e chunk, salvando poi il database vettoriale 
        # nel path del file system definito in configurazione
        vectorIndex.add(chunks, vectors)
        vectorIndex.save(self.configuration.vectorStore.path)

        # Incapsulo i metadati dello step in un oggetto utile per le lavorazioni successive
        return IndexingReport(
            documentCount=len(documents),
            chunkCount=len(chunks),
            vectorDimension=vectorIndex.getDimension(),
            elapsedTimeMs=(self.clock() - startTime) * 1000,
            outputPath=self.configuration.vectorStore.path,
        )

    def createEmbeddingModel(self) -> EmbeddingModel:
        embeddingConfiguration = self.configuration.embeddings
        return SentenceTransformerEmbeddingModel(
            modelName=embeddingConfiguration.model,
            batchSize=embeddingConfiguration.batchSize,
            normalizeEmbeddings=embeddingConfiguration.normalize,
            inputPrefix=embeddingConfiguration.passagePrefix,
        )

    @staticmethod
    def validateContent(
        documents: list[Document],
        chunks: list[DocumentChunk],
    ) -> None:
        if not documents:
            raise ValueError("La Knowledge Base non contiene documenti supportati.")
        if not chunks:
            raise ValueError("La Knowledge Base non ha prodotto chunk indicizzabili.")
