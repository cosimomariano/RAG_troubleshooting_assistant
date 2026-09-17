from collections.abc import Sequence

import pytest

from app.indexing import EmbeddingModel, SentenceTransformerEmbeddingModel


class FakeSentenceTransformer:
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
        if self.vectors is not None:
            return self.vectors
        return [[len(text), index] for index, text in enumerate(sentences)]


def test_adapter_implements_embedding_model_protocol() -> None:
    adapter = SentenceTransformerEmbeddingModel(
        "synthetic-embedding-model",
        backend=FakeSentenceTransformer(),
    )

    assert isinstance(adapter, EmbeddingModel)
    assert adapter.model_name == "synthetic-embedding-model"


def test_texts_are_encoded_in_one_configured_batch() -> None:
    backend = FakeSentenceTransformer()
    adapter = SentenceTransformerEmbeddingModel(
        "synthetic-embedding-model",
        batch_size=2,
        normalize_embeddings=True,
        backend=backend,
    )

    vectors = adapter.encode(["errore gateway", "timeout database"])

    assert vectors == [[14.0, 0.0], [16.0, 1.0]]
    assert backend.calls == [
        {
            "sentences": ["errore gateway", "timeout database"],
            "batch_size": 2,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
    ]


def test_normalization_option_is_forwarded_to_backend() -> None:
    backend = FakeSentenceTransformer()
    adapter = SentenceTransformerEmbeddingModel(
        "synthetic-embedding-model",
        normalize_embeddings=False,
        backend=backend,
    )

    adapter.encode(["testo sintetico"])

    assert backend.calls[0]["normalize_embeddings"] is False


def test_empty_batch_does_not_call_backend() -> None:
    backend = FakeSentenceTransformer()
    adapter = SentenceTransformerEmbeddingModel(
        "synthetic-embedding-model",
        backend=backend,
    )

    assert adapter.encode([]) == []
    assert backend.calls == []


@pytest.mark.parametrize(
    ("model_name", "batch_size"),
    [("", 32), ("   ", 32), ("synthetic-embedding-model", 0), ("model", -1)],
)
def test_invalid_configuration_is_rejected(model_name: str, batch_size: int) -> None:
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddingModel(
            model_name,
            batch_size=batch_size,
            backend=FakeSentenceTransformer(),
        )


def test_backend_must_return_one_vector_per_text() -> None:
    adapter = SentenceTransformerEmbeddingModel(
        "synthetic-embedding-model",
        backend=FakeSentenceTransformer(vectors=[[1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="un embedding per ogni testo"):
        adapter.encode(["primo testo", "secondo testo"])
