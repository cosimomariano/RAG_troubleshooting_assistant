import re

from pydantic import BaseModel, ConfigDict


def convertCamelCaseToSnakeCase(fieldName: str) -> str:
    """Converte il nome Java del campo nel nome usato dal contratto JSON/YAML."""

    firstPass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", fieldName)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", firstPass).lower()


class StrictModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=convertCamelCaseToSnakeCase,
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )
