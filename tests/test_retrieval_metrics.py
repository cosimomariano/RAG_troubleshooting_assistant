import pytest
from app.evaluation import RetrievalMetricsCalculator

def test_recall_at_k_counts_relevant_documents_in_first_positions() -> None:
    retrieved_documents = [
        "architecture/system-overview.md",
        "runbooks/payment-failure.md",
        "services/payment-service.md",
        "errors/error-catalog.md",
    ]
    relevant_documents = {
        "runbooks/payment-failure.md",
        "errors/error-catalog.md",
    }

    recall_at_three = RetrievalMetricsCalculator.recall_at_k(
        retrieved_documents,
        relevant_documents,
        k=3,
    )
    recall_at_four = RetrievalMetricsCalculator.recall_at_k(
        retrieved_documents,
        relevant_documents,
        k=4,
    )

    assert recall_at_three == 0.5
    assert recall_at_four == 1.0

def test_recall_at_k_counts_a_document_only_once() -> None:
    retrieved_documents = [
        "runbooks/payment-failure.md",
        "runbooks/payment-failure.md",
    ]

    recall = RetrievalMetricsCalculator.recall_at_k(
        retrieved_documents,
        {
            "runbooks/payment-failure.md",
            "errors/error-catalog.md",
        },
        k=2,
    )

    assert recall == 0.5

def test_reciprocal_rank_uses_first_relevant_position() -> None:
    reciprocal_rank = RetrievalMetricsCalculator.reciprocal_rank(
        [
            "architecture/system-overview.md",
            "services/checkout-service.md",
            "runbooks/payment-failure.md",
            "errors/error-catalog.md",
        ],
        {
            "runbooks/payment-failure.md",
            "errors/error-catalog.md",
        },
    )

    assert reciprocal_rank == pytest.approx(1 / 3)

def test_reciprocal_rank_is_zero_when_no_relevant_document_is_retrieved() -> None:
    reciprocal_rank = RetrievalMetricsCalculator.reciprocal_rank(
        ["architecture/system-overview.md"],
        {"runbooks/payment-failure.md"},
    )

    assert reciprocal_rank == 0.0

def test_mean_reciprocal_rank_averages_all_cases() -> None:
    mean_reciprocal_rank = RetrievalMetricsCalculator.mean_reciprocal_rank(
        retrieved_rankings=[
            ["runbooks/payment-failure.md"],
            ["architecture/system-overview.md", "runbooks/cart-failure.md"],
            ["services/checkout-service.md"],
        ],
        relevant_documents_by_case=[
            {"runbooks/payment-failure.md"},
            {"runbooks/cart-failure.md"},
            {"runbooks/product-catalog-failure.md"},
        ],
    )

    assert mean_reciprocal_rank == 0.5

@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError, match="maggiore di zero"):
        RetrievalMetricsCalculator.recall_at_k(
            ["runbooks/payment-failure.md"],
            {"runbooks/payment-failure.md"},
            k,
        )

def test_metrics_reject_empty_relevant_documents() -> None:
    with pytest.raises(ValueError, match="almeno un documento rilevante"):
        RetrievalMetricsCalculator.reciprocal_rank(
            ["runbooks/payment-failure.md"],
            set(),
        )

def test_mean_reciprocal_rank_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="almeno una graduatoria"):
        RetrievalMetricsCalculator.mean_reciprocal_rank([], [])

def test_mean_reciprocal_rank_requires_one_relevance_set_per_ranking() -> None:
    with pytest.raises(ValueError, match="numero di graduatorie"):
        RetrievalMetricsCalculator.mean_reciprocal_rank(
            [["runbooks/payment-failure.md"]],
            [],
        )