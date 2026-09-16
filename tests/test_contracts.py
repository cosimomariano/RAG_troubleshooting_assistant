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

def test_openapi_contract_is_valid() -> None:
    specification = load_yaml("contracts/openapi/troubleshooting-api.yaml")

    validate(specification, cls=OpenAPIV31SpecValidator)

def test_base_application_configuration_loads() -> None:
    configuration = load_yaml("configs/application.yaml")

    assert set(configuration) == { "application", "server", "logging", "llm", "ingestion", "chunking"}
    assert configuration["application"]["name"] == "rag-troubleshooting-assistant"
    assert configuration["ingestion"]["supported_extensions"] == [".md", ".txt"]
    assert configuration["chunking"]["chunk_overlap_characters"] < configuration["chunking"]["chunk_size_characters"]

def test_environment_placeholders_are_documented() -> None:
    configuration = (PROJECT_ROOT / "configs/application.yaml").read_text(encoding="utf-8")
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    configured_variables = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)\}", configuration))
    documented_variables = {
        line.partition("=")[0].strip()
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert configured_variables == documented_variables