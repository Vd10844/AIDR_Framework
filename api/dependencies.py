from infrastructure.pipeline.document_pipeline import DocumentPipeline
from application.services.extraction_service import ExtractionService


def get_extraction_service():
    pipeline = DocumentPipeline("config/config.yaml")
    return ExtractionService(extractor=pipeline)
