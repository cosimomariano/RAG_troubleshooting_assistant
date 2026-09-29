from argparse import ArgumentParser

from app.bootstrap import KnowledgeBaseIndexer
from app.cli.common import (
    addApplicationConfigurationArguments,
    loadApplicationConfiguration,
)


def buildArgumentParser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Costruisce l'indice vettoriale della Knowledge Base.",
    )
    addApplicationConfigurationArguments(parser)
    return parser


def main() -> None:
    arguments = buildArgumentParser().parse_args()
    configuration = loadApplicationConfiguration(arguments)
    report = KnowledgeBaseIndexer(configuration).build()

    print("Indicizzazione completata.")
    print(f"Documenti caricati: {report.documentCount}")
    print(f"Chunk indicizzati: {report.chunkCount}")
    print(f"Dimensione dei vettori: {report.vectorDimension}")
    print(f"Tempo impiegato: {report.elapsedTimeMs:.2f} ms")
    print(f"Indice salvato in: {report.outputPath}")


if __name__ == "__main__":
    main()
