from pathlib import Path

from pydantic import Field, ValidationError

from app.config.yaml_reader import YamlObjectReader
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

    def __init__(self, yamlObjectReader: YamlObjectReader | None = None) -> None:
        self.yamlObjectReader = yamlObjectReader or YamlObjectReader()

    def load(self, configurationPath: str | Path) -> ExperimentConfiguration:
        serializedConfiguration = self.yamlObjectReader.read(
            configurationPath,
            "sperimentale",
        )

        try:
            return ExperimentConfiguration.model_validate(serializedConfiguration)
        except ValidationError as error:
            raise ValueError("La configurazione sperimentale non rispetta lo schema.") from error
