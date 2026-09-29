import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from app.config.yaml_reader import YamlObjectReader
from app.models import StrictModel
from app.retrieval import RetrievalMode

ENVIRONMENT_REFERENCE_PATTERN = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


class ApplicationIdentityConfiguration(StrictModel):
    name: str = Field(min_length=1, description="Nome dell'applicazione")
    environment: str = Field(min_length=1, description="Ambiente di esecuzione")


class ServerConfiguration(StrictModel):
    host: str = Field(min_length=1, description="Indirizzo di ascolto del server API")
    port: int = Field(ge=1, le=65535, description="Porta di ascolto del server API")


class LoggingConfiguration(StrictModel):
    level: str = Field(min_length=1, description="Livello minimo dei log applicativi")
    format: Literal["json", "text"] = Field(description="Formato dei log applicativi")


class LLMConfiguration(StrictModel):
    provider: Literal["ollama"] = Field(description="Provider del modello generativo")
    baseUrl: str = Field(min_length=1, description="URL di base del server Ollama remoto")
    model: str = Field(min_length=1, description="Nome del modello esposto da Ollama")
    timeoutSeconds: float = Field(gt=0, description="Timeout della chiamata LLM in secondi")


class IngestionConfiguration(StrictModel):
    knowledgeBasePath: Path = Field(description="Percorso della Knowledge Base")
    knowledgeBaseVersion: str = Field(
        min_length=1,
        description="Versione logica della Knowledge Base",
    )
    supportedExtensions: tuple[str, ...] = Field(
        min_length=1,
        description="Estensioni documentali ammesse",
    )
    recursive: bool = Field(description="Abilita la scansione ricorsiva delle fonti")


class ChunkingConfiguration(StrictModel):
    strategy: Literal["section_aware"] = Field(description="Strategia di chunking")
    chunkSizeCharacters: int = Field(
        gt=0,
        description="Dimensione massima del chunk in caratteri",
    )
    chunkOverlapCharacters: int = Field(
        ge=0,
        description="Sovrapposizione tra chunk consecutivi in caratteri",
    )


class MaskingConfiguration(StrictModel):
    enabled: bool = Field(description="Abilita il mascheramento dei dati sensibili")
    categories: tuple[str, ...] = Field(
        description="Categorie di dati sensibili censite",
    )


class EmbeddingConfiguration(StrictModel):
    provider: Literal["sentence_transformers"] = Field(
        description="Provider usato per calcolare gli embedding",
    )
    model: str = Field(min_length=1, description="Nome del modello bi-encoder")
    batchSize: int = Field(gt=0, description="Numero di testi elaborati per batch")
    normalize: bool = Field(description="Abilita la normalizzazione degli embedding")


class VectorStoreConfiguration(StrictModel):
    provider: Literal["faiss"] = Field(description="Provider dell'indice vettoriale")
    path: Path = Field(description="Percorso di persistenza dell'indice FAISS")
    metric: Literal["inner_product"] = Field(description="Metrica di similarità vettoriale")


class RetrievalConfiguration(StrictModel):
    mode: RetrievalMode = Field(description="Modalità di retrieval da eseguire")
    topK: int = Field(gt=0, description="Numero massimo di fonti restituite")


class RerankerConfiguration(StrictModel):
    enabled: bool = Field(description="Abilita il secondo stadio di reranking")
    provider: Literal["cross_encoder"] = Field(description="Provider del reranker")
    model: str = Field(min_length=1, description="Nome del modello Cross-Encoder")
    batchSize: int = Field(gt=0, description="Numero di coppie elaborate per batch")
    candidateTopN: int = Field(
        gt=0,
        description="Numero di candidati inviati al Cross-Encoder",
    )


class EvaluationConfiguration(StrictModel):
    datasetPath: Path = Field(description="Percorso del golden dataset")
    resultsPath: Path = Field(description="Directory dei risultati sperimentali")


class ApplicationConfiguration(StrictModel):
    application: ApplicationIdentityConfiguration
    server: ServerConfiguration
    logging: LoggingConfiguration
    llm: LLMConfiguration
    ingestion: IngestionConfiguration
    chunking: ChunkingConfiguration
    masking: MaskingConfiguration
    embeddings: EmbeddingConfiguration
    vectorStore: VectorStoreConfiguration
    retrieval: RetrievalConfiguration
    reranker: RerankerConfiguration
    evaluation: EvaluationConfiguration


class EnvironmentFileLoader:
    """Legge un file .env senza modificare le variabili del processo."""

    def load(self, environmentFile: str | Path) -> dict[str, str]:
        path = Path(environmentFile)
        if not path.is_file():
            raise FileNotFoundError(f"File delle variabili di ambiente non trovato: {path}")

        variables: dict[str, str] = {}
        for lineNumber, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            parsedEntry = self.parseLine(line, lineNumber)
            if parsedEntry is None:
                continue
            name, value = parsedEntry
            variables[name] = value
        return variables

    @staticmethod
    def parseLine(line: str, lineNumber: int) -> tuple[str, str] | None:
        normalizedLine = line.strip()
        if not normalizedLine or normalizedLine.startswith("#"):
            return None

        if normalizedLine.startswith("export "):
            normalizedLine = normalizedLine.removeprefix("export ").strip()
        if "=" not in normalizedLine:
            raise ValueError(f"Riga {lineNumber} del file .env non valida.")

        name, value = normalizedLine.split("=", maxsplit=1)
        normalizedName = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalizedName):
            raise ValueError(f"Nome non valido alla riga {lineNumber} del file .env.")

        normalizedValue = value.strip()
        if len(normalizedValue) >= 2 and normalizedValue[0] == normalizedValue[-1]:
            if normalizedValue[0] in {"'", '"'}:
                normalizedValue = normalizedValue[1:-1]
        return normalizedName, normalizedValue


class ApplicationConfigurationLoader:
    """Carica il contratto YAML e risolve i riferimenti alle variabili d'ambiente."""

    def __init__(
        self,
        environmentFileLoader: EnvironmentFileLoader | None = None,
        yamlObjectReader: YamlObjectReader | None = None,
    ) -> None:
        self.environmentFileLoader = environmentFileLoader or EnvironmentFileLoader()
        self.yamlObjectReader = yamlObjectReader or YamlObjectReader()

    def load(
        self,
        configurationPath: str | Path,
        environmentFile: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> ApplicationConfiguration:
        serializedConfiguration = self.yamlObjectReader.read(
            configurationPath,
            "applicativa",
        )
        variables = self.loadVariables(environmentFile, environment)
        resolvedConfiguration = self.resolveValue(serializedConfiguration, variables)

        try:
            return ApplicationConfiguration.model_validate(resolvedConfiguration)
        except ValidationError as error:
            raise ValueError("La configurazione applicativa non rispetta lo schema.") from error

    def loadVariables(
        self,
        environmentFile: str | Path | None,
        environment: Mapping[str, str] | None,
    ) -> dict[str, str]:
        variables: dict[str, str] = {}
        if environmentFile is not None:
            variables.update(self.environmentFileLoader.load(environmentFile))

        processEnvironment = os.environ if environment is None else environment
        variables.update(processEnvironment)
        return variables

    def resolveValue(self, value: Any, variables: Mapping[str, str]) -> Any:
        if isinstance(value, dict):
            return {
                key: self.resolveValue(nestedValue, variables) for key, nestedValue in value.items()
            }
        if isinstance(value, list):
            return [self.resolveValue(item, variables) for item in value]
        if not isinstance(value, str):
            return value

        referenceMatch = ENVIRONMENT_REFERENCE_PATTERN.fullmatch(value)
        if referenceMatch is None:
            return value

        variableName = referenceMatch.group(1)
        variableValue = variables.get(variableName)
        if variableValue is None or not variableValue.strip():
            raise ValueError(f"La variabile d'ambiente '{variableName}' non è valorizzata.")
        return variableValue
