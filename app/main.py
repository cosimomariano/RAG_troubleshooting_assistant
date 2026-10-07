from pathlib import Path

import uvicorn
from fastapi import FastAPI

from app.bootstrap import RAGApplicationFactory
from app.config import ApplicationConfiguration, ApplicationConfigurationLoader

DEFAULT_APPLICATION_CONFIGURATION = Path("configs/application.yaml")
DEFAULT_ENVIRONMENT_FILE = Path(".env")


def loadConfiguration() -> ApplicationConfiguration:
    return ApplicationConfigurationLoader().load(
        configurationPath=DEFAULT_APPLICATION_CONFIGURATION,
        environmentFile=DEFAULT_ENVIRONMENT_FILE,
    )


def createApplication(configuration: ApplicationConfiguration) -> FastAPI:
    return RAGApplicationFactory(configuration).createApi()


def main() -> None:
    # Caricamento della configurazione (application.yml e .env)
    configuration = loadConfiguration()

    # Creazione API /troubleshoot sulla base della configurazione caricata
    application = createApplication(configuration)

    # Esposizione son uvicorn dell'endpoint
    uvicorn.run(
        application,
        host=configuration.server.host,
        port=configuration.server.port,
        log_level=configuration.logging.level.casefold(),
    )


if __name__ == "__main__":
    main()
