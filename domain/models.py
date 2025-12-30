# This file used to define domain models and data structures
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ExtractionResult:
    status: str
    kvps: Dict[str, str]
    confidences: Dict[str, float]
    model_version: str
    processing_ms: int
