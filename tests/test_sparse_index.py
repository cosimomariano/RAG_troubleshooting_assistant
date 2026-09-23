import pytest
from app.indexing import BM25SparseIndex, SparseIndex, TechnicalTextTokenizer
from app.models import DocumentChunk, SourceMetadata

def build_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="runbook-checkout",
        text=text,
        metadata=SourceMetadata(
            source="knowledge_base/runbooks/checkout.md",
            document_type="runbook",
            service="checkout",
            section="diagnosi",
        ),
    )


def build_sparse_index() -> BM25SparseIndex:
    return BM25SparseIndex(
        [
            build_chunk(
                "payment-unreachable",
                "La chiamata payment/charge fallisce con connection refused.",
            ),
            build_chunk(
                "cart-empty",
                "Il carrello non contiene prodotti disponibili.",
            ),
            build_chunk(
                "payment-timeout",
                "Il servizio payment supera il timeout configurato.",
            ),
        ]
    )


def test_bm25_index_satisfies_sparse_index_contract() -> None:
    sparse_index = build_sparse_index()

    assert isinstance(sparse_index, SparseIndex)
    assert sparse_index.size == 3


def test_technical_tokenizer_preserves_identifiers_and_api_paths() -> None:
    tokenizer = TechnicalTextTokenizer()

    tokens = tokenizer.tokenize("Errore ERR_PAYMENT_TIMEOUT su Payment/Charge.")

    assert tokens == ["errore", "err_payment_timeout", "su", "payment/charge"]


def test_search_ranks_exact_technical_terms_first() -> None:
    sparse_index = build_sparse_index()

    matches = sparse_index.search("connection refused payment/charge", k=2)

    assert [position for position, _ in matches] == [0, 1]
    assert matches[0][1] > matches[1][1]
    assert sparse_index.get_chunk(matches[0][0]).id == "payment-unreachable"


def test_top_k_larger_than_corpus_returns_only_available_chunks() -> None:
    sparse_index = build_sparse_index()

    matches = sparse_index.search("timeout", k=10)

    assert len(matches) == 3


def test_empty_corpus_returns_no_results() -> None:
    sparse_index = BM25SparseIndex([])

    assert sparse_index.search("payment timeout", k=3) == []


@pytest.mark.parametrize("k", [0, -1])
def test_search_result_count_must_be_positive(k: int) -> None:
    sparse_index = build_sparse_index()

    with pytest.raises(ValueError, match="maggiore di zero"):
        sparse_index.search("payment", k=k)


def test_duplicate_chunk_ids_are_rejected() -> None:
    first_chunk = build_chunk("payment", "Connection refused verso payment.")
    duplicate_chunk = build_chunk("payment", "Timeout del servizio payment.")

    with pytest.raises(ValueError, match="duplicati"):
        BM25SparseIndex([first_chunk, duplicate_chunk])


def test_chunk_without_searchable_terms_is_rejected() -> None:
    chunk = build_chunk("invalid", "...")

    with pytest.raises(ValueError, match="non contiene termini"):
        BM25SparseIndex([chunk])