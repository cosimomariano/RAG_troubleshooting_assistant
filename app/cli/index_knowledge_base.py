from argparse import ArgumentParser

from app.bootstrap import KnowledgeBaseIndexer
from app.cli.common import (
    add_application_configuration_arguments,
    load_application_configuration,
)


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Costruisce l'indice vettoriale della Knowledge Base.",
    )
    add_application_configuration_arguments(parser)
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    configuration = load_application_configuration(arguments)
    report = KnowledgeBaseIndexer(configuration).build()

    print("Indicizzazione completata.")
    print(f"Documenti caricati: {report.document_count}")
    print(f"Chunk indicizzati: {report.chunk_count}")
    print(f"Dimensione dei vettori: {report.vector_dimension}")
    print(f"Tempo impiegato: {report.elapsed_time_ms:.2f} ms")
    print(f"Indice salvato in: {report.output_path}")


if __name__ == "__main__":
    main()
