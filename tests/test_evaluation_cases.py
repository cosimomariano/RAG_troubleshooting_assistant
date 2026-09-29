import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation import CaseDifficulty, GoldenCase, GoldenCaseLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DATASET_PATH = PROJECT_ROOT / "data" / "golden_dataset" / "cases.jsonl"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "knowledge_base"


def testProjectGoldenDatasetContainsValidCuratedCases() -> None:
    cases = GoldenCaseLoader().load(GOLDEN_DATASET_PATH)

    assert len(cases) == 5
    assert len({case.id for case in cases}) == len(cases)
    assert {case.difficulty for case in cases} >= {
        CaseDifficulty.EXACT,
        CaseDifficulty.PARAPHRASE,
        CaseDifficulty.DISTRIBUTED,
    }

    for case in cases:
        for relativePath in case.relevantDocuments:
            assert (KNOWLEDGE_BASE_PATH / relativePath).is_file()


def testCaseNormalizesTextAndDocumentPaths() -> None:
    case = GoldenCase(
        id="  case_payment  ",
        question="  Perché il pagamento fallisce?  ",
        incidentContext="  PaymentService.Charge status=ERROR  ",
        expectedAnswer="  Verificare Payment e il feature flag.  ",
        relevantDocuments=[" runbooks\\payment-failure.md "],
        service="  payment  ",
        expectedRootCause="   ",
        difficulty=CaseDifficulty.STACKTRACE,
    )

    assert case.id == "case_payment"
    assert case.question == "Perché il pagamento fallisce?"
    assert case.relevantDocuments == ("runbooks/payment-failure.md",)
    assert case.service == "payment"
    assert case.expectedRootCause is None


def testCaseRejectsDuplicateRelevantDocuments() -> None:
    with pytest.raises(ValidationError, match="duplicati"):
        GoldenCase(
            id="case_duplicate_documents",
            question="Qual è la causa?",
            incidentContext="PaymentService.Charge status=ERROR",
            expectedAnswer="Verificare il servizio Payment.",
            relevantDocuments=[
                "runbooks/payment-failure.md",
                "runbooks/payment-failure.md",
            ],
        )


def testLoaderRejectsDuplicateCaseIds(tmp_path: Path) -> None:
    datasetPath = tmp_path / "duplicate-cases.jsonl"
    serializedCase = json.dumps(
        {
            "id": "case_duplicate",
            "question": "Qual è la causa?",
            "incident_context": "CartService.EmptyCart status=ERROR",
            "expected_answer": "Verificare CartService.EmptyCart.",
            "relevant_documents": ["runbooks/cart-failure.md"],
        }
    )
    datasetPath.write_text(f"{serializedCase}\n{serializedCase}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Identificativo del caso duplicato"):
        GoldenCaseLoader().load(datasetPath)


def testLoaderReportsTheInvalidLineNumber(tmp_path: Path) -> None:
    datasetPath = tmp_path / "invalid-case.jsonl"
    datasetPath.write_text('\n{"id": "missing-fields"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="riga 2"):
        GoldenCaseLoader().load(datasetPath)
