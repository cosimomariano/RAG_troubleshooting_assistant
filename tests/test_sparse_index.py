import pytest

from app.indexing import BM25SparseIndex, SparseIndex, TechnicalTextTokenizer
from app.models import DocumentChunk, SourceMetadata


def buildChunk(chunkId: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunkId,
        documentId="runbook-checkout",
        text=text,
        metadata=SourceMetadata(
            source="knowledge_base/runbooks/checkout.md",
            documentType="runbook",
            service="checkout",
            section="diagnosi",
        ),
    )


def buildSparseIndex() -> BM25SparseIndex:
    return BM25SparseIndex(
        [
            buildChunk(
                "payment-unreachable",
                "La chiamata payment/charge fallisce con connection refused.",
            ),
            buildChunk(
                "cart-empty",
                "Il carrello non contiene prodotti disponibili.",
            ),
            buildChunk(
                "payment-timeout",
                "Il servizio payment supera il timeout configurato.",
            ),
        ]
    )


def testBm25IndexSatisfiesSparseIndexContract() -> None:
    sparseIndex = buildSparseIndex()

    assert isinstance(sparseIndex, SparseIndex)
    assert sparseIndex.getSize() == 3


def testTechnicalTokenizerPreservesIdentifiersAndApiPaths() -> None:
    tokenizer = TechnicalTextTokenizer()

    tokens = tokenizer.tokenize("Errore ERR_PAYMENT_TIMEOUT su Payment/Charge.")

    assert tokens == ["errore", "err_payment_timeout", "su", "payment/charge"]


def testSearchRanksExactTechnicalTermsFirst() -> None:
    sparseIndex = buildSparseIndex()

    matches = sparseIndex.search("connection refused payment/charge", k=2)

    assert [position for position, _ in matches] == [0, 1]
    assert matches[0][1] > matches[1][1]
    assert sparseIndex.getChunk(matches[0][0]).id == "payment-unreachable"


def testTopKLargerThanCorpusReturnsOnlyAvailableChunks() -> None:
    sparseIndex = buildSparseIndex()

    matches = sparseIndex.search("timeout", k=10)

    assert len(matches) == 3


def testEmptyCorpusReturnsNoResults() -> None:
    sparseIndex = BM25SparseIndex([])

    assert sparseIndex.search("payment timeout", k=3) == []


@pytest.mark.parametrize("k", [0, -1])
def testSearchResultCountMustBePositive(k: int) -> None:
    sparseIndex = buildSparseIndex()

    with pytest.raises(ValueError, match="maggiore di zero"):
        sparseIndex.search("payment", k=k)


def testDuplicateChunkIdsAreRejected() -> None:
    firstChunk = buildChunk("payment", "Connection refused verso payment.")
    duplicateChunk = buildChunk("payment", "Timeout del servizio payment.")

    with pytest.raises(ValueError, match="duplicati"):
        BM25SparseIndex([firstChunk, duplicateChunk])


def testChunkWithoutSearchableTermsIsRejected() -> None:
    chunk = buildChunk("invalid", "...")

    with pytest.raises(ValueError, match="non contiene termini"):
        BM25SparseIndex([chunk])
