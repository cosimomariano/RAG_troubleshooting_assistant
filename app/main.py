from pathlib import Path

import uvicorn
from fastapi import FastAPI

from app.bootstrap import RAGApplicationFactory
from app.config import ApplicationConfiguration, ApplicationConfigurationLoader

DEFAULT_APPLICATION_CONFIGURATION = Path("configs/application.yaml")
DEFAULT_ENVIRONMENT_FILE = Path(".env")


def load_configuration() -> ApplicationConfiguration:
    return ApplicationConfigurationLoader().load(
        configuration_path=DEFAULT_APPLICATION_CONFIGURATION,
        environment_file=DEFAULT_ENVIRONMENT_FILE,
    )


def create_application(configuration: ApplicationConfiguration) -> FastAPI:
    return RAGApplicationFactory(configuration).create_api()


def main() -> None:
    configuration = load_configuration()
    application = create_application(configuration)
    uvicorn.run(
        application,
        host=configuration.server.host,
        port=configuration.server.port,
        log_level=configuration.logging.level.casefold(),
    )


if __name__ == "__main__":
    main()
