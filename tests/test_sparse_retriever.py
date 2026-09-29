from app.indexing import BM25SparseIndex
from app.models import DocumentChunk, SourceMetadata
from app.retrieval import Retriever, SparseRetriever


def buildChunk(chunkId: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunkId,
        documentId="runbook-payment",
        text=text,
        metadata=SourceMetadata(
            source="knowledge_base/runbooks/payment.md",
            documentType="runbook",
            service="payment",
            section="diagnosi",
        ),
    )


def buildSparseRetriever() -> SparseRetriever:
    sparseIndex = BM25SparseIndex(
        [
            buildChunk(
                "payment-unreachable",
                "Connection refused durante la chiamata payment/charge.",
            ),
            buildChunk(
                "cart-empty",
                "Il carrello non contiene prodotti.",
            ),
            buildChunk(
                "payment-timeout",
                "Timeout del servizio payment.",
            ),
        ]
    )
    return SparseRetriever(sparseIndex)


def testSparseRetrieverSatisfiesCommonRetrieverContract() -> None:
    retriever = buildSparseRetriever()

    assert isinstance(retriever, Retriever)


def testSparseRetrieverReturnsCommonRetrievalResults() -> None:
    retriever = buildSparseRetriever()

    results = retriever.retrieve("connection refused payment/charge", k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "payment-unreachable"
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score is not None
    assert {result.retriever for result in results} == {"sparse"}
