import re
from pathlib import Path

import yaml
from openapi_spec_validator import OpenAPIV31SpecValidator, validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def loadYaml(relativePath: str) -> dict:
    content = (PROJECT_ROOT / relativePath).read_text(encoding="utf-8")
    document = yaml.safe_load(content)

    assert isinstance(document, dict)
    return document


def testTroubleshootingOpenapiIsAValidVersion31Contract() -> None:
    specification = loadYaml("contracts/openapi/troubleshooting-api.yaml")

    validate(specification, cls=OpenAPIV31SpecValidator)


def testConfigurationExposesOnlyComponentsAvailableAtThisStage() -> None:
    configuration = loadYaml("configs/application.yaml")

    assert set(configuration) == {
        "application",
        "server",
        "logging",
        "llm",
        "ingestion",
        "chunking",
        "masking",
        "embeddings",
        "vector_store",
        "retrieval",
        "reranker",
        "evaluation",
    }
    assert configuration["application"]["name"] == "rag-troubleshooting-assistant"
    assert configuration["ingestion"]["supported_extensions"] == [".md", ".txt"]
    assert configuration["ingestion"]["knowledge_base_version"] == ("${KNOWLEDGE_BASE_VERSION}")
    assert (
        configuration["chunking"]["chunk_overlap_characters"]
        < configuration["chunking"]["chunk_size_characters"]
    )
    assert configuration["masking"] == {
        "enabled": True,
        "categories": ["credentials", "email", "iban", "private_ipv4"],
    }
    assert configuration["embeddings"] == {
        "provider": "sentence_transformers",
        "model": "${EMBEDDING_MODEL}",
        "batch_size": 32,
        "normalize": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    }
    assert configuration["vector_store"] == {
        "provider": "faiss",
        "path": "${VECTOR_STORE_PATH}",
        "metric": "inner_product",
    }
    assert configuration["retrieval"] == {
        "mode": "${RETRIEVAL_MODE}",
        "top_k": "${RETRIEVAL_TOP_K}",
    }
    assert configuration["reranker"] == {
        "enabled": "${RERANKER_ENABLED}",
        "provider": "cross_encoder",
        "model": "${RERANKER_MODEL}",
        "batch_size": "${RERANKER_BATCH_SIZE}",
        "candidate_top_n": "${RERANKER_CANDIDATE_TOP_N}",
    }
    assert configuration["evaluation"] == {
        "dataset_path": "${GOLDEN_DATASET_PATH}",
        "results_path": "${EXPERIMENT_RESULTS_PATH}",
    }


def testEveryConfigurationPlaceholderHasAnEnvExampleEntry() -> None:
    configuration = (PROJECT_ROOT / "configs/application.yaml").read_text(encoding="utf-8")
    envExample = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    configuredVariables = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", configuration))
    documentedVariables = {
        line.partition("=")[0].strip()
        for line in envExample.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert configuredVariables == documentedVariables
