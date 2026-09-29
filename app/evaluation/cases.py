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
    incidentContext: str = Field(
        min_length=1,
        description="Evidenze sintetiche o raccolte relative all'incidente",
    )
    expectedAnswer: str = Field(
        min_length=1,
        description="Contenuto minimo atteso nella diagnosi",
    )
    relevantDocuments: tuple[str, ...] = Field(
        min_length=1,
        description="Percorsi relativi dei documenti rilevanti nella Knowledge Base",
    )
    service: str | None = Field(
        default=None,
        description="Microservizio principalmente interessato",
    )
    expectedRootCause: str | None = Field(
        default=None,
        description="Causa radice attesa per il caso",
    )
    difficulty: CaseDifficulty | None = Field(
        default=None,
        description="Classe di difficoltà del caso sperimentale",
    )

    @field_validator("id", "question", "incidentContext", "expectedAnswer", mode="before")
    @classmethod
    def normalizeRequiredText(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("service", "expectedRootCause", mode="before")
    @classmethod
    def normalizeOptionalText(cls, value: object) -> object:
        if isinstance(value, str):
            normalizedValue = value.strip()
            return normalizedValue or None
        return value

    @field_validator("relevantDocuments", mode="before")
    @classmethod
    def normalizeRelevantDocuments(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            return value

        normalizedDocuments: list[str] = []
        for document in value:
            if not isinstance(document, str) or not document.strip():
                raise ValueError("Ogni documento rilevante deve avere un percorso non vuoto.")
            normalizedDocuments.append(document.strip().replace("\\", "/"))

        if len(normalizedDocuments) != len(set(normalizedDocuments)):
            raise ValueError("I documenti rilevanti non possono contenere duplicati.")
        return tuple(normalizedDocuments)


class GoldenCaseLoader:
    """Carica e valida un golden dataset nel formato JSON Lines."""

    def load(self, datasetPath: str | Path) -> list[GoldenCase]:
        path = Path(datasetPath)
        self.validatePath(path)

        cases: list[GoldenCase] = []
        caseIds: set[str] = set()

        for lineNumber, rawLine in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            serializedCase = rawLine.strip()
            if not serializedCase:
                continue

            case = self.parseCase(serializedCase, lineNumber)
            if case.id in caseIds:
                raise ValueError(f"Identificativo del caso duplicato: {case.id}")

            caseIds.add(case.id)
            cases.append(case)

        if not cases:
            raise ValueError("Il golden dataset non contiene casi di troubleshooting.")
        return cases

    @staticmethod
    def validatePath(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Golden dataset non trovato: {path}")

    @staticmethod
    def parseCase(serializedCase: str, lineNumber: int) -> GoldenCase:
        try:
            return GoldenCase.model_validate_json(serializedCase)
        except ValidationError as error:
            raise ValueError(
                f"Caso del golden dataset non valido alla riga {lineNumber}."
            ) from error
