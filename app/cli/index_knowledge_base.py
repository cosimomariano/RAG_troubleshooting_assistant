from argparse import ArgumentParser

from app.bootstrap import KnowledgeBaseIndexer
from app.cli.common import (
    addApplicationConfigurationArguments,
    loadApplicationConfiguration,
)

# Entry point pipeline offline

def buildArgumentParser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Costruisce l'indice vettoriale della Knowledge Base.",
    )
    addApplicationConfigurationArguments(parser)
    return parser


def main() -> None:
    arguments = buildArgumentParser().parse_args()
    # Leggo la configurazione: application.yml e .env
    configuration = loadApplicationConfiguration(arguments)
    # Avvio l'indicizzazione della knowledge base sulla base della configurazione fornita
    report = KnowledgeBaseIndexer(configuration).build()

    print("Indicizzazione completata.")
    print(f"Documenti caricati: {report.documentCount}")
    print(f"Chunk indicizzati: {report.chunkCount}")
    print(f"Dimensione dei vettori: {report.vectorDimension}")
    print(f"Tempo impiegato: {report.elapsedTimeMs:.2f} ms")
    print(f"Indice salvato in: {report.outputPath}")


if __name__ == "__main__":
    main()
