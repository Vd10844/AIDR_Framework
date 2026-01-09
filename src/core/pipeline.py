import yaml
import io
import uuid
from typing import List, Tuple
from PIL import Image
import pypdfium2 as pdfium
from src.core.model_interface import ModelInterface, ExtractionResult
from src.core.audit_logger import audit_logger
from src.adapters.storage import hitl_storage
from loguru import logger
from time import time
from src.monitoring.metrics import (
    REQUESTS_TOTAL,
    REQUEST_FAILURES_TOTAL,
    REQUEST_LATENCY_MS,
    HITL_RATE,
    CONFIDENCE_MEAN,
    MISSING_KEYS_TOTAL
)

class DocumentPipeline:
    def __init__(self, config_path: str):
        logger.info(f"Initializing pipeline with config: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.model = self._create_model()
    
    def _create_model(self):
        model_config = self.config['model']
        model_name = model_config['name']
        logger.info(f"Loading model: {model_name}")
        
        if model_name == "PP-StructureV3":
            from src.models.pp_structure import PPStructureV3
            return PPStructureV3()
        raise ValueError(f"Model {model_name} not supported")
    
    async def _preprocess(self, content: bytes, file_name: str) -> List[Image.Image]:
        """Convert PDF or generic image bytes to a list of PIL Images."""
        if file_name.lower().endswith('.pdf'):
            pdf = pdfium.PdfDocument(content)
            images = []
            for page in pdf:
                bitmap = page.render()
                pil_image = bitmap.to_pil()
                images.append(pil_image)
            return images
        
        # Assume it's an image
        image = Image.open(io.BytesIO(content))
        return [image]

    async def extract(self, content: bytes, file_name: str) -> Tuple[ExtractionResult, bool]:
        request_id = str(uuid.uuid4())
        logger.info(f"Processing request {request_id} for file {file_name}")

        REQUESTS_TOTAL.inc()
        start = time()

        try:
            # -------------------------
            # Preprocess
            # -------------------------
            images = await self._preprocess(content, file_name)

            image_bytes_list = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_bytes_list.append(buf.getvalue())

            # -------------------------
            # Model extraction
            # -------------------------
            result = await self.model.extract(
                image_bytes_list,
                self.config["model"]["keys"]
            )

            # -------------------------
            # Metrics: confidence & coverage
            # -------------------------
            if result.confidences:
                mean_conf = sum(result.confidences.values()) / len(result.confidences)
            else:
                mean_conf = 0.0

            CONFIDENCE_MEAN.set(mean_conf)

            for key, value in result.kvps.items():
                if value == "Not Found":
                    MISSING_KEYS_TOTAL.labels(key=key).inc()

            # -------------------------
            # HITL decision
            # -------------------------
            hitl_required = any(c < 0.8 for c in result.confidences.values())
            HITL_RATE.set(1.0 if hitl_required else 0.0)

            if hitl_required:
                logger.warning(f"Low confidence detected for {request_id}. Routing to HITL.")
                hitl_path = hitl_storage.save_for_review(
                    file_name, content, request_id
                )
                logger.info(f"File saved for manual review at: {hitl_path}")

            # -------------------------
            # Audit log
            # -------------------------
            audit_logger.log_extraction(
                request_id=request_id,
                file_name=file_name,
                result={
                    "kvps": result.kvps,
                    "confidences": result.confidences
                },
                status="review" if hitl_required else "approved"
            )

            return result, hitl_required

        except Exception as e:
            REQUEST_FAILURES_TOTAL.inc()
            logger.error(f"Extraction failed for {request_id}: {str(e)}")
            raise

        finally:
            REQUEST_LATENCY_MS.observe((time() - start) * 1000)
