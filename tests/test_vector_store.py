import json
from pathlib import Path

import pytest

from app.indexing import FaissVectorIndex, PersistentVectorIndex
from app.models import DocumentChunk, SourceMetadata


def build_chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="runbook-checkout",
        text=text,
        metadata=SourceMetadata(
            source="docs/runbooks/checkout.md",
            document_type="runbook",
            service="checkout",
            section="diagnosi",
        ),
    )


def test_faiss_index_satisfies_persistent_vector_index_contract() -> None:
    index = FaissVectorIndex(dimension=3)

    assert isinstance(index, PersistentVectorIndex)
    assert index.dimension == 3
    assert index.size == 0


def test_added_vectors_keep_the_same_positions_as_their_chunks() -> None:
    index = FaissVectorIndex(dimension=3)
    symptom = build_chunk(
        "checkout-symptom",
        "Il servizio checkout restituisce HTTP 500.",
    )
    action = build_chunk(
        "checkout-action",
        "Controllare la raggiungibilità del servizio payment.",
    )

    index.add(
        [symptom, action],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    assert index.size == 2
    assert index.get_chunk(0) == symptom
    assert index.get_chunk(1) == action


def test_empty_batch_does_not_change_the_index() -> None:
    index = FaissVectorIndex(dimension=3)
    index.add([], [])

    assert index.size == 0


def test_chunk_and_vector_counts_must_match() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = build_chunk("checkout-symptom", "Timeout verso il servizio payment.")

    with pytest.raises(ValueError, match="numero di chunk"):
        index.add([chunk], [])


def test_vector_dimension_must_match_the_faiss_index() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = build_chunk("checkout-symptom", "Timeout verso il servizio payment.")

    with pytest.raises(ValueError, match="dimensione 3"):
        index.add([chunk], [[1.0, 0.0]])


def test_duplicate_chunk_ids_are_rejected_before_indexing() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = build_chunk("checkout-symptom", "Timeout verso il servizio payment.")
    duplicate = build_chunk("checkout-symptom", "Errore di connessione a payment.")

    with pytest.raises(ValueError, match="duplicati"):
        index.add(
            [chunk, duplicate],
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        )

    assert index.size == 0


def test_a_chunk_cannot_be_added_twice_in_separate_batches() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = build_chunk("checkout-symptom", "Timeout verso il servizio payment.")
    index.add([chunk], [[1.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="già presenti"):
        index.add([chunk], [[1.0, 0.0, 0.0]])

    assert index.size == 1


def test_get_chunk_rejects_positions_outside_the_index() -> None:
    index = FaissVectorIndex(dimension=3)

    with pytest.raises(IndexError):
        index.get_chunk(0)
    with pytest.raises(IndexError):
        index.get_chunk(-1)


def test_search_returns_positions_ordered_by_inner_product() -> None:
    index = FaissVectorIndex(dimension=3)
    chunks = [
        build_chunk("checkout-timeout", "Timeout durante il pagamento."),
        build_chunk("cart-empty", "Il carrello risulta vuoto."),
        build_chunk("payment-unreachable", "Il servizio payment non è raggiungibile."),
    ]
    index.add(
        chunks,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]],
    )

    matches = index.search([1.0, 0.0, 0.0], k=2)

    assert [position for position, _ in matches] == [0, 2]
    assert [score for _, score in matches] == pytest.approx([1.0, 0.8])


def test_search_never_returns_empty_faiss_positions() -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = build_chunk("checkout-timeout", "Timeout durante il pagamento.")
    index.add([chunk], [[1.0, 0.0, 0.0]])

    assert index.search([1.0, 0.0, 0.0], k=5) == [(0, 1.0)]


@pytest.mark.parametrize("k", [0, -1])
def test_search_result_count_must_be_positive(k: int) -> None:
    index = FaissVectorIndex(dimension=3)

    with pytest.raises(ValueError, match="maggiore di zero"):
        index.search([1.0, 0.0, 0.0], k=k)


def test_saved_index_can_be_loaded_without_losing_chunk_mapping(
    tmp_path: Path,
) -> None:
    index = FaissVectorIndex(dimension=3)
    chunks = [
        build_chunk("checkout-symptom", "Il checkout restituisce HTTP 500."),
        build_chunk("checkout-action", "Verificare il servizio payment."),
    ]
    index.add(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    index.save(tmp_path)
    restored = FaissVectorIndex.load(tmp_path)

    assert (tmp_path / "dense.index").is_file()
    assert (tmp_path / "chunks.json").is_file()
    assert restored.dimension == 3
    assert restored.size == 2
    assert [restored.get_chunk(position) for position in range(restored.size)] == chunks


def test_load_rejects_a_chunk_mapping_not_aligned_with_faiss(
    tmp_path: Path,
) -> None:
    index = FaissVectorIndex(dimension=3)
    chunk = build_chunk("checkout-symptom", "Il checkout restituisce HTTP 500.")
    index.add([chunk], [[1.0, 0.0, 0.0]])
    index.save(tmp_path)

    mapping_path = tmp_path / "chunks.json"
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    payload["chunks"] = []
    mapping_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="quantità diverse"):
        FaissVectorIndex.load(tmp_path)


@pytest.mark.parametrize("dimension", [0, -1])
def test_index_dimension_must_be_positive(dimension: int) -> None:
    with pytest.raises(ValueError, match="maggiore di zero"):
        FaissVectorIndex(dimension=dimension)
