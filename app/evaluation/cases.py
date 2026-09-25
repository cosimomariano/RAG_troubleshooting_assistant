from enum import StrEnum
from pathlib import Path
from pydantic import Field, ValidationError, field_validator
from app.models.base import StrictModel

class CaseDifficulty(StrEnum):
    """Difficoltà usata per segmentare i risultati sperimentali."""

    EXACT = "exact"
    STACKTRACE = "stacktrace"
    PARAPHRASE = "paraphrase"
    DISTRIBUTED = "distributed"

class GoldenCase(StrictModel):
    id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Identificativo stabile del caso sperimentale",
    )
    question: str = Field(min_length=1, description="Domanda sottoposta al sistema")
    incident_context: str = Field(
        min_length=1,
        description="Evidenze sintetiche o raccolte relative all'incidente",
    )
    expected_answer: str = Field(
        min_length=1,
        description="Contenuto minimo atteso nella diagnosi",
    )
    relevant_documents: tuple[str, ...] = Field(
        min_length=1,
        description="Percorsi relativi dei documenti rilevanti nella Knowledge Base",
    )
    service: str | None = Field(
        default=None,
        description="Microservizio principalmente interessato",
    )
    expected_root_cause: str | None = Field(
        default=None,
        description="Causa radice attesa per il caso",
    )
    difficulty: CaseDifficulty | None = Field(
        default=None,
        description="Classe di difficoltà del caso sperimentale",
    )

    @field_validator("id", "question", "incident_context", "expected_answer", mode="before")
    @classmethod
    def normalize_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("service", "expected_root_cause", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized_value = value.strip()
            return normalized_value or None
        return value

    @field_validator("relevant_documents", mode="before")
    @classmethod
    def normalize_relevant_documents(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value

        normalized_documents: list[str] = []
        for document in value:
            if not isinstance(document, str) or not document.strip():
                raise ValueError("Ogni documento rilevante deve avere un percorso non vuoto.")
            normalized_documents.append(document.strip().replace("\\", "/"))

        if len(normalized_documents) != len(set(normalized_documents)):
            raise ValueError("I documenti rilevanti non possono contenere duplicati.")
        return tuple(normalized_documents)


class GoldenCaseLoader:
    """Carica e valida un golden dataset nel formato JSON Lines."""

    def load(self, dataset_path: str | Path) -> list[GoldenCase]:
        path = Path(dataset_path)
        self._validate_path(path)

        cases: list[GoldenCase] = []
        case_ids: set[str] = set()

        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            serialized_case = raw_line.strip()
            if not serialized_case:
                continue

            case = self._parse_case(serialized_case, line_number)
            if case.id in case_ids:
                raise ValueError(f"Identificativo del caso duplicato: {case.id}")

            case_ids.add(case.id)
            cases.append(case)

        if not cases:
            raise ValueError("Il golden dataset non contiene casi di troubleshooting.")
        return cases

    @staticmethod
    def _validate_path(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Golden dataset non trovato: {path}")

    @staticmethod
    def _parse_case(serialized_case: str, line_number: int) -> GoldenCase:
        try:
            return GoldenCase.model_validate_json(serialized_case)
        except ValidationError as error:
            raise ValueError(
                f"Caso del golden dataset non valido alla riga {line_number}."
            ) from error