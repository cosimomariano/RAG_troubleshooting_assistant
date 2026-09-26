from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from app.models import StrictModel
from app.retrieval import RetrievalMode


class ExperimentDescriptor(StrictModel):
    id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Identificativo stabile della configurazione sperimentale",
    )
    description: str = Field(
        min_length=1,
        description="Descrizione della configurazione sperimentale",
    )


class RetrievalExperimentConfiguration(StrictModel):
    mode: RetrievalMode = Field(description="Modalità di retrieval sottoposta a valutazione")


class RerankerExperimentConfiguration(StrictModel):
    enabled: bool = Field(
        default=False,
        description="Indica se il secondo stadio di reranking è abilitato",
    )


class ExperimentConfiguration(StrictModel):
    experiment: ExperimentDescriptor
    retrieval: RetrievalExperimentConfiguration
    reranker: RerankerExperimentConfiguration = Field(
        default_factory=RerankerExperimentConfiguration
    )


class ExperimentConfigurationLoader:
    """Carica e valida una configurazione sperimentale YAML."""

    def load(self, configuration_path: str | Path) -> ExperimentConfiguration:
        path = Path(configuration_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configurazione sperimentale non trovata: {path}")

        try:
            serialized_configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            raise ValueError("La configurazione sperimentale contiene YAML non valido.") from error

        if not isinstance(serialized_configuration, dict):
            raise ValueError("La configurazione sperimentale deve essere un oggetto YAML.")

        try:
            return ExperimentConfiguration.model_validate(serialized_configuration)
        except ValidationError as error:
            raise ValueError("La configurazione sperimentale non rispetta lo schema.") from error
