import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from app.evaluation import CaseDifficulty, GoldenCase, GoldenCaseLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DATASET_PATH = PROJECT_ROOT / "data" / "golden_dataset" / "cases.jsonl"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "knowledge_base"

def test_project_golden_dataset_contains_valid_curated_cases() -> None:
    cases = GoldenCaseLoader().load(GOLDEN_DATASET_PATH)

    assert len(cases) == 5
    assert len({case.id for case in cases}) == len(cases)
    assert {case.difficulty for case in cases} >= {
        CaseDifficulty.EXACT,
        CaseDifficulty.PARAPHRASE,
        CaseDifficulty.DISTRIBUTED,
    }

    for case in cases:
        for relative_path in case.relevant_documents:
            assert (KNOWLEDGE_BASE_PATH / relative_path).is_file()

def test_case_normalizes_text_and_document_paths() -> None:
    case = GoldenCase(
        id="  case_payment  ",
        question="  Perché il pagamento fallisce?  ",
        incident_context="  PaymentService.Charge status=ERROR  ",
        expected_answer="  Verificare Payment e il feature flag.  ",
        relevant_documents=[" runbooks\\payment-failure.md "],
        service="  payment  ",
        expected_root_cause="   ",
        difficulty=CaseDifficulty.STACKTRACE,
    )

    assert case.id == "case_payment"
    assert case.question == "Perché il pagamento fallisce?"
    assert case.relevant_documents == ("runbooks/payment-failure.md",)
    assert case.service == "payment"
    assert case.expected_root_cause is None

def test_case_rejects_duplicate_relevant_documents() -> None:
    with pytest.raises(ValidationError, match="duplicati"):
        GoldenCase(
            id="case_duplicate_documents",
            question="Qual è la causa?",
            incident_context="PaymentService.Charge status=ERROR",
            expected_answer="Verificare il servizio Payment.",
            relevant_documents=[
                "runbooks/payment-failure.md",
                "runbooks/payment-failure.md",
            ],
        )

def test_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "duplicate-cases.jsonl"
    serialized_case = json.dumps(
        {
            "id": "case_duplicate",
            "question": "Qual è la causa?",
            "incident_context": "CartService.EmptyCart status=ERROR",
            "expected_answer": "Verificare CartService.EmptyCart.",
            "relevant_documents": ["runbooks/cart-failure.md"],
        }
    )
    dataset_path.write_text(f"{serialized_case}\n{serialized_case}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Identificativo del caso duplicato"):
        GoldenCaseLoader().load(dataset_path)

def test_loader_reports_the_invalid_line_number(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid-case.jsonl"
    dataset_path.write_text('\n{"id": "missing-fields"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="riga 2"):
        GoldenCaseLoader().load(dataset_path)