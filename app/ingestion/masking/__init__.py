"""Rilevamento e mascheramento dei dati sensibili prima dell'indicizzazione."""

from app.ingestion.masking.base import SensitiveDataMasker
from app.ingestion.masking.regex import MaskingRule, RegexSensitiveDataMasker

__all__ = ["MaskingRule", "RegexSensitiveDataMasker", "SensitiveDataMasker"]
