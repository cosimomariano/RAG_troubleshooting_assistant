import json
from pathlib import Path

import pytest

from app.indexing import FaissVectorIndex, PersistentVectorIndex
from app.models import DocumentChunk, SourceMetadata


def buildChunk(chunkId: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunkId,
        documentId="runbook-checkout",
        text=text,
        metadata=SourceMetadata(
            source="docs/runbooks/checkout.md",
            documentType="runbook",
            service="checkout",
            section="diagnosi",
        ),
    )


def testFaissIndexSatisfiesPersistentVectorIndexContract() -> None:
    index = FaissVectorIndex(dimension=3)

    assert isinstance(index, PersistentVectorIndex)
    assert index.getDimension() == 3
    assert index.getSize() == 0


def testAddedVectorsKeepTheSamePositionsAsTheirChunks() -> None:
    index = FaissVectorIndex(dimension=3)
    symptom = buildChunk(
        "checkout-symptom",
        "Il servizio checkout restituisce HTTP 500.",
    )
    action = buildChunk(
        "checkout-action",
        "Controllare la raggiungibilità del servizio payment.",
    )

    index.add(
        [symptom, action],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    assert index.getSize() == 2
    assert index.getChunk(0) == symptom
    assert index.getChunk(1) == action


def testEmptyBatchDoesNotChangeTheIndex() -> None:
    index = FaissVectorIndex(dimension=3)
    index.add([], [])

    assert index.getSize() == 0


def testChunkAndVectorCountsMustMatch() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = buildChunk("checkout-symptom", "Timeout verso il servizio payment.")

    with pytest.raises(ValueError, match="numero di chunk"):
        index.add([chunk], [])


def testVectorDimensionMustMatchTheFaissIndex() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = buildChunk("checkout-symptom", "Timeout verso il servizio payment.")

    with pytest.raises(ValueError, match="dimensione 3"):
        index.add([chunk], [[1.0, 0.0]])


def testDuplicateChunkIdsAreRejectedBeforeIndexing() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = buildChunk("checkout-symptom", "Timeout verso il servizio payment.")
    duplicate = buildChunk("checkout-symptom", "Errore di connessione a payment.")

    with pytest.raises(ValueError, match="duplicati"):
        index.add(
            [chunk, duplicate],
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        )

    assert index.getSize() == 0


def testAChunkCannotBeAddedTwiceInSeparateBatches() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = buildChunk("checkout-symptom", "Timeout verso il servizio payment.")
    index.add([chunk], [[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="già presenti"):
        index.add([chunk], [[1.0, 0.0, 0.0]])

    assert index.getSize() == 1


def testGetChunkRejectsPositionsOutsideTheIndex() -> None:
    index = FaissVectorIndex(dimension=3)

    with pytest.raises(IndexError):
        index.getChunk(0)
    with pytest.raises(IndexError):
        index.getChunk(-1)


def testSearchReturnsPositionsOrderedByInnerProduct() -> None:
    index = FaissVectorIndex(dimension=3)
    chunks = [
        buildChunk("checkout-timeout", "Timeout durante il pagamento."),
        buildChunk("cart-empty", "Il carrello risulta vuoto."),
        buildChunk("payment-unreachable", "Il servizio payment non è raggiungibile."),
    ]
    index.add(
        chunks,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]],
    )

    matches = index.search([1.0, 0.0, 0.0], k=2)

    assert [position for position, _ in matches] == [0, 2]
    assert [score for _, score in matches] == pytest.approx([1.0, 0.8])


def testSearchNeverReturnsEmptyFaissPositions() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = buildChunk("checkout-timeout", "Timeout durante il pagamento.")
    index.add([chunk], [[1.0, 0.0, 0.0]])

    assert index.search([1.0, 0.0, 0.0], k=5) == [(0, 1.0)]


@pytest.mark.parametrize("k", [0, -1])
def testSearchResultCountMustBePositive(k: int) -> None:
    index = FaissVectorIndex(dimension=3)

    with pytest.raises(ValueError, match="maggiore di zero"):
        index.search([1.0, 0.0, 0.0], k=k)


def testSavedIndexCanBeLoadedWithoutLosingChunkMapping(
    tmp_path: Path,
) -> None:
    index = FaissVectorIndex(dimension=3)
    chunks = [
        buildChunk("checkout-symptom", "Il checkout restituisce HTTP 500."),
        buildChunk("checkout-action", "Verificare il servizio payment."),
    ]
    index.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    index.save(tmp_path)
    restored = FaissVectorIndex.load(tmp_path)

    assert (tmp_path / "dense.index").is_file()
    assert (tmp_path / "chunks.json").is_file()
    assert restored.getDimension() == 3
    assert restored.getSize() == 2
    assert [restored.getChunk(position) for position in range(restored.getSize())] == chunks


def testLoadRejectsAChunkMappingNotAlignedWithFaiss(
    tmp_path: Path,
) -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = buildChunk("checkout-symptom", "Il checkout restituisce HTTP 500.")
    index.add([chunk], [[1.0, 0.0, 0.0]])
    index.save(tmp_path)

    mappingPath = tmp_path / "chunks.json"
    payload = json.loads(mappingPath.read_text(encoding="utf-8"))
    payload["chunks"] = []
    mappingPath.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="quantità diverse"):
        FaissVectorIndex.load(tmp_path)


@pytest.mark.parametrize("dimension", [0, -1])
def testIndexDimensionMustBePositive(dimension: int) -> None:
    with pytest.raises(ValueError, match="maggiore di zero"):
        FaissVectorIndex(dimension=dimension)
