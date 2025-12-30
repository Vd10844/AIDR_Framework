from application.ports.extractor import Extractor
from domain.models import ExtractionResult


class DocumentPipeline(Extractor):

    def __init__(self, config_path: str):
        # load models, configs, thresholds
        self.config_path = config_path
        self.model_version = "PP-StructureV2"

    async def extract(
        self,
        content: bytes,
        file_name: str,
        request_id: str
    ) -> ExtractionResult:
        # actual OCR logic (use sync to worker if heavy)
        # ... return ExtractionResult ...
        return ExtractionResult(
            status="SUCCESS",
            kvps={},
            confidences={},
            model_version=self.model_version,
            processing_ms=0
        )