from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.config import ApplicationConfiguration, ApplicationConfigurationLoader

DEFAULT_APPLICATION_CONFIGURATION = Path("configs/application.yaml")
DEFAULT_ENVIRONMENT_FILE = Path(".env")


def add_application_configuration_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--application-config",
        type=Path,
        default=DEFAULT_APPLICATION_CONFIGURATION,
        help="Percorso del file YAML di configurazione applicativa.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENVIRONMENT_FILE,
        help="Percorso del file contenente le variabili d'ambiente.",
    )


def load_application_configuration(arguments: Namespace) -> ApplicationConfiguration:
    return ApplicationConfigurationLoader().load(
        configuration_path=arguments.application_config,
        environment_file=arguments.env_file,
    )
