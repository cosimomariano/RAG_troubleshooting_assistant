import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError

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
    base_url: str = Field(min_length=1, description="URL di base del server Ollama remoto")
    model: str = Field(min_length=1, description="Nome del modello esposto da Ollama")
    timeout_seconds: float = Field(gt=0, description="Timeout della chiamata LLM in secondi")


class IngestionConfiguration(StrictModel):
    knowledge_base_path: Path = Field(description="Percorso della Knowledge Base")
    knowledge_base_version: str = Field(
        min_length=1,
        description="Versione logica della Knowledge Base",
    )
    supported_extensions: tuple[str, ...] = Field(
        min_length=1,
        description="Estensioni documentali ammesse",
    )
    recursive: bool = Field(description="Abilita la scansione ricorsiva delle fonti")


class ChunkingConfiguration(StrictModel):
    strategy: Literal["section_aware"] = Field(description="Strategia di chunking")
    chunk_size_characters: int = Field(
        gt=0,
        description="Dimensione massima del chunk in caratteri",
    )
    chunk_overlap_characters: int = Field(
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
    batch_size: int = Field(gt=0, description="Numero di testi elaborati per batch")
    normalize: bool = Field(description="Abilita la normalizzazione degli embedding")


class VectorStoreConfiguration(StrictModel):
    provider: Literal["faiss"] = Field(description="Provider dell'indice vettoriale")
    path: Path = Field(description="Percorso di persistenza dell'indice FAISS")
    metric: Literal["inner_product"] = Field(description="Metrica di similarità vettoriale")


class RetrievalConfiguration(StrictModel):
    mode: RetrievalMode = Field(description="Modalità di retrieval da eseguire")
    top_k: int = Field(gt=0, description="Numero massimo di fonti restituite")


class RerankerConfiguration(StrictModel):
    enabled: bool = Field(description="Abilita il secondo stadio di reranking")
    provider: Literal["cross_encoder"] = Field(description="Provider del reranker")
    model: str = Field(min_length=1, description="Nome del modello Cross-Encoder")
    batch_size: int = Field(gt=0, description="Numero di coppie elaborate per batch")
    candidate_top_n: int = Field(
        gt=0,
        description="Numero di candidati inviati al Cross-Encoder",
    )


class EvaluationConfiguration(StrictModel):
    dataset_path: Path = Field(description="Percorso del golden dataset")
    results_path: Path = Field(description="Directory dei risultati sperimentali")


class ApplicationConfiguration(StrictModel):
    application: ApplicationIdentityConfiguration
    server: ServerConfiguration
    logging: LoggingConfiguration
    llm: LLMConfiguration
    ingestion: IngestionConfiguration
    chunking: ChunkingConfiguration
    masking: MaskingConfiguration
    embeddings: EmbeddingConfiguration
    vector_store: VectorStoreConfiguration
    retrieval: RetrievalConfiguration
    reranker: RerankerConfiguration
    evaluation: EvaluationConfiguration


class EnvironmentFileLoader:
    """Legge un file .env senza modificare le variabili del processo."""

    def load(self, environment_file: str | Path) -> dict[str, str]:
        path = Path(environment_file)
        if not path.is_file():
            raise FileNotFoundError(f"File delle variabili di ambiente non trovato: {path}")

        variables: dict[str, str] = {}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            parsed_entry = self._parse_line(line, line_number)
            if parsed_entry is None:
                continue
            name, value = parsed_entry
            variables[name] = value
        return variables

    @staticmethod
    def _parse_line(line: str, line_number: int) -> tuple[str, str] | None:
        normalized_line = line.strip()
        if not normalized_line or normalized_line.startswith("#"):
            return None

        if normalized_line.startswith("export "):
            normalized_line = normalized_line.removeprefix("export ").strip()
        if "=" not in normalized_line:
            raise ValueError(f"Riga {line_number} del file .env non valida.")

        name, value = normalized_line.split("=", maxsplit=1)
        normalized_name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized_name):
            raise ValueError(f"Nome non valido alla riga {line_number} del file .env.")

        normalized_value = value.strip()
        if len(normalized_value) >= 2 and normalized_value[0] == normalized_value[-1]:
            if normalized_value[0] in {"'", '"'}:
                normalized_value = normalized_value[1:-1]
        return normalized_name, normalized_value


class ApplicationConfigurationLoader:
    """Carica il contratto YAML e risolve i riferimenti alle variabili d'ambiente."""

    def __init__(self, environment_file_loader: EnvironmentFileLoader | None = None) -> None:
        self._environment_file_loader = environment_file_loader or EnvironmentFileLoader()

    def load(
        self,
        configuration_path: str | Path,
        environment_file: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> ApplicationConfiguration:
        path = Path(configuration_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configurazione applicativa non trovata: {path}")

        serialized_configuration = self._read_yaml(path)
        variables = self._load_variables(environment_file, environment)
        resolved_configuration = self._resolve_value(serialized_configuration, variables)

        try:
            return ApplicationConfiguration.model_validate(resolved_configuration)
        except ValidationError as error:
            raise ValueError("La configurazione applicativa non rispetta lo schema.") from error

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        try:
            serialized_configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError("La configurazione applicativa contiene YAML non valido.") from error

        if not isinstance(serialized_configuration, dict):
            raise ValueError("La configurazione applicativa deve essere un oggetto YAML.")
        return serialized_configuration

    def _load_variables(
        self,
        environment_file: str | Path | None,
        environment: Mapping[str, str] | None,
    ) -> dict[str, str]:
        variables: dict[str, str] = {}
        if environment_file is not None:
            variables.update(self._environment_file_loader.load(environment_file))

        process_environment = os.environ if environment is None else environment
        variables.update(process_environment)
        return variables

    def _resolve_value(self, value: Any, variables: Mapping[str, str]) -> Any:
        if isinstance(value, dict):
            return {
                key: self._resolve_value(nested_value, variables)
                for key, nested_value in value.items()
            }
        if isinstance(value, list):
            return [self._resolve_value(item, variables) for item in value]
        if not isinstance(value, str):
            return value

        reference_match = ENVIRONMENT_REFERENCE_PATTERN.fullmatch(value)
        if reference_match is None:
            return value

        variable_name = reference_match.group(1)
        variable_value = variables.get(variable_name)
        if variable_value is None or not variable_value.strip():
            raise ValueError(
                f"La variabile d'ambiente '{variable_name}' non è valorizzata."
            )
        return variable_value
