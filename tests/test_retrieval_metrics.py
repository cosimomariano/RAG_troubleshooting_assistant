import pytest

from app.evaluation import RetrievalMetricsCalculator


def testRecallAtKCountsRelevantDocumentsInFirstPositions() -> None:
    retrievedDocuments = [
        "architecture/system-overview.md",
        "runbooks/payment-failure.md",
        "services/payment-service.md",
        "errors/error-catalog.md",
    ]
    relevantDocuments = {
        "runbooks/payment-failure.md",
        "errors/error-catalog.md",
    }

    recallAtThree = RetrievalMetricsCalculator.recallAtK(
        retrievedDocuments,
        relevantDocuments,
        k=3,
    )
    recallAtFour = RetrievalMetricsCalculator.recallAtK(
        retrievedDocuments,
        relevantDocuments,
        k=4,
    )

    assert recallAtThree == 0.5
    assert recallAtFour == 1.0


def testRecallAtKCountsADocumentOnlyOnce() -> None:
    retrievedDocuments = [
        "runbooks/payment-failure.md",
        "runbooks/payment-failure.md",
    ]

    recall = RetrievalMetricsCalculator.recallAtK(
        retrievedDocuments,
        {
            "runbooks/payment-failure.md",
            "errors/error-catalog.md",
        },
        k=2,
    )

    assert recall == 0.5


def testReciprocalRankUsesFirstRelevantPosition() -> None:
    reciprocalRank = RetrievalMetricsCalculator.reciprocalRank(
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

    assert reciprocalRank == pytest.approx(1 / 3)


def testReciprocalRankIsZeroWhenNoRelevantDocumentIsRetrieved() -> None:
    reciprocalRank = RetrievalMetricsCalculator.reciprocalRank(
        ["architecture/system-overview.md"],
        {"runbooks/payment-failure.md"},
    )

    assert reciprocalRank == 0.0


def testMeanReciprocalRankAveragesAllCases() -> None:
    meanReciprocalRank = RetrievalMetricsCalculator.meanReciprocalRank(
        retrievedRankings=[
            ["runbooks/payment-failure.md"],
            ["architecture/system-overview.md", "runbooks/cart-failure.md"],
            ["services/checkout-service.md"],
        ],
        relevantDocumentsByCase=[
            {"runbooks/payment-failure.md"},
            {"runbooks/cart-failure.md"},
            {"runbooks/product-catalog-failure.md"},
        ],
    )

    assert meanReciprocalRank == 0.5


@pytest.mark.parametrize("k", [0, -1])
def testRecallAtKRejectsNonPositiveK(k: int) -> None:
    with pytest.raises(ValueError, match="maggiore di zero"):
        RetrievalMetricsCalculator.recallAtK(
            ["runbooks/payment-failure.md"],
            {"runbooks/payment-failure.md"},
            k,
        )


def testMetricsRejectEmptyRelevantDocuments() -> None:
    with pytest.raises(ValueError, match="almeno un documento rilevante"):
        RetrievalMetricsCalculator.reciprocalRank(
            ["runbooks/payment-failure.md"],
            set(),
        )


def testMeanReciprocalRankRejectsEmptyDataset() -> None:
    with pytest.raises(ValueError, match="almeno una graduatoria"):
        RetrievalMetricsCalculator.meanReciprocalRank([], [])


def testMeanReciprocalRankRequiresOneRelevanceSetPerRanking() -> None:
    with pytest.raises(ValueError, match="numero di graduatorie"):
        RetrievalMetricsCalculator.meanReciprocalRank(
            [["runbooks/payment-failure.md"]],
            [],
        )
