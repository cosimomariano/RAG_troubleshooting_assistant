from collections.abc import Sequence

import pytest

from app.indexing import EmbeddingModel, SentenceTransformerEmbeddingModel


class RecordingEncoder:
    """Backend leggero: registra la chiamata senza caricare un modello reale."""

    def __init__(self, vectors: Sequence[Sequence[float]] | None = None) -> None:
        self.vectors = vectors
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Sequence[Sequence[float]]:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        return self.vectors if self.vectors is not None else []


def test_sentence_transformer_adapter_respects_embedding_contract() -> None:
    model = SentenceTransformerEmbeddingModel(
        "test-bi-encoder",
        backend=RecordingEncoder(),
    )

    assert isinstance(model, EmbeddingModel)
    assert model.model_name == "test-bi-encoder"


def test_runbook_chunks_are_sent_to_backend_as_a_single_batch() -> None:
    expected_vectors = [[0.12, 0.98], [0.87, 0.14]]
    backend = RecordingEncoder(vectors=expected_vectors)
    model = SentenceTransformerEmbeddingModel(
        "test-bi-encoder",
        batch_size=8,
        normalize_embeddings=True,
        backend=backend,
    )
    chunks = [
        "Il servizio checkout non raggiunge payment.",
        "Verificare endpoint e disponibilità del servizio payment.",
    ]

    vectors = model.encode(chunks)

    assert vectors == expected_vectors
    assert backend.calls == [
        {
            "sentences": chunks,
            "batch_size": 8,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
    ]


def test_normalization_can_be_disabled_for_embedding_experiments() -> None:
    backend = RecordingEncoder(vectors=[[0.4, 0.6]])
    model = SentenceTransformerEmbeddingModel(
        "test-bi-encoder",
        normalize_embeddings=False,
        backend=backend,
    )

    model.encode(["Errore HTTP 500 restituito dal servizio cart."])

    assert backend.calls[0]["normalize_embeddings"] is False


def test_empty_chunk_collection_skips_model_inference() -> None:
    backend = RecordingEncoder()
    model = SentenceTransformerEmbeddingModel(
        "test-bi-encoder",
        backend=backend,
    )

    assert model.encode([]) == []
    assert backend.calls == []


@pytest.mark.parametrize(
    ("model_name", "batch_size"),
    [
        pytest.param("", 32, id="nome-modello-vuoto"),
        pytest.param("   ", 32, id="nome-modello-con-soli-spazi"),
        pytest.param("test-bi-encoder", 0, id="batch-size-zero"),
        pytest.param("test-bi-encoder", -1, id="batch-size-negativo"),
    ],
)
def test_invalid_embedding_configuration_is_rejected(
    model_name: str,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddingModel(
            model_name,
            batch_size=batch_size,
            backend=RecordingEncoder(),
        )


def test_backend_cannot_drop_a_chunk_from_the_batch() -> None:
    backend = RecordingEncoder(vectors=[[0.2, 0.8]])
    model = SentenceTransformerEmbeddingModel(
        "test-bi-encoder",
        backend=backend,
    )

    with pytest.raises(ValueError):
        model.encode(
            [
                "Sintomo osservato nel servizio checkout.",
                "Verifica suggerita per il servizio payment.",
            ]
        )
