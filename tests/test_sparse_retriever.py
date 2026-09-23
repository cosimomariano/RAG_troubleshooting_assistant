from app.indexing import BM25SparseIndex
from app.models import DocumentChunk, SourceMetadata
from app.retrieval import Retriever, SparseRetriever

def build_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="runbook-payment",
        text=text,
        metadata=SourceMetadata(
            source="knowledge_base/runbooks/payment.md",
            document_type="runbook",
            service="payment",
            section="diagnosi",
        ),
    )

def build_sparse_retriever() -> SparseRetriever:
    sparse_index = BM25SparseIndex(
        [
            build_chunk(
                "payment-unreachable",
                "Connection refused durante la chiamata payment/charge.",
            ),
            build_chunk(
                "cart-empty",
                "Il carrello non contiene prodotti.",
            ),
            build_chunk(
                "payment-timeout",
                "Timeout del servizio payment.",
            ),
        ]
    )
    return SparseRetriever(sparse_index)

def test_sparse_retriever_satisfies_common_retriever_contract() -> None:
    retriever = build_sparse_retriever()

    assert isinstance(retriever, Retriever)


def test_sparse_retriever_returns_common_retrieval_results() -> None:
    retriever = build_sparse_retriever()

    results = retriever.retrieve("connection refused payment/charge", k=2)

    assert len(results) == 2
    assert results[0].chunk.id == "payment-unreachable"
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score is not None
    assert {result.retriever for result in results} == {"sparse"}