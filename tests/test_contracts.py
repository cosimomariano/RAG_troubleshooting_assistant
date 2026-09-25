import re
from pathlib import Path

import yaml
from openapi_spec_validator import OpenAPIV31SpecValidator, validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(relative_path: str) -> dict:
    content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    document = yaml.safe_load(content)

    assert isinstance(document, dict)
    return document


def test_troubleshooting_openapi_is_a_valid_version_31_contract() -> None:
    specification = load_yaml("contracts/openapi/troubleshooting-api.yaml")

    validate(specification, cls=OpenAPIV31SpecValidator)


def test_configuration_exposes_only_components_available_at_this_stage() -> None:
    configuration = load_yaml("configs/application.yaml")

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
    }
    assert configuration["application"]["name"] == "rag-troubleshooting-assistant"
    assert configuration["ingestion"]["supported_extensions"] == [".md", ".txt"]
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
        "enabled": False,
        "provider": "cross_encoder",
        "model": "${RERANKER_MODEL}",
        "batch_size": "${RERANKER_BATCH_SIZE}",
        "candidate_top_n": "${RERANKER_CANDIDATE_TOP_N}",
    }


def test_every_configuration_placeholder_has_an_env_example_entry() -> None:
    configuration = (PROJECT_ROOT / "configs/application.yaml").read_text(encoding="utf-8")
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    configured_variables = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", configuration))
    documented_variables = {
        line.partition("=")[0].strip()
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert configured_variables == documented_variables
