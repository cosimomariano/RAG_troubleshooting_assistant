from argparse import ArgumentParser, Namespace
from pathlib import Path

from app.config import ApplicationConfiguration, ApplicationConfigurationLoader

DEFAULT_APPLICATION_CONFIGURATION = Path("configs/application.yaml")
DEFAULT_ENVIRONMENT_FILE = Path(".env")


def addApplicationConfigurationArguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--application-config",
        dest="applicationConfig",
        type=Path,
        default=DEFAULT_APPLICATION_CONFIGURATION,
        help="Percorso del file YAML di configurazione applicativa.",
    )
    parser.add_argument(
        "--env-file",
        dest="environmentFile",
        type=Path,
        default=DEFAULT_ENVIRONMENT_FILE,
        help="Percorso del file contenente le variabili d'ambiente.",
    )


def loadApplicationConfiguration(arguments: Namespace) -> ApplicationConfiguration:
    return ApplicationConfigurationLoader().load(
        configurationPath=arguments.applicationConfig,
        environmentFile=arguments.environmentFile,
    )
