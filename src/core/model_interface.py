from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ExtractionResult:
    kvps: Dict[str, str]
    confidences: Dict[str, float]
    status: str  # approved | review

class ModelInterface(ABC):
    @abstractmethod
    async def extract(self, images: List[bytes], keys: List[str]) -> ExtractionResult:
<<<<<<< Updated upstream
        pass
=======
        pass
>>>>>>> Stashed changes
