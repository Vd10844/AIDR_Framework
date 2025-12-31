from src.core.model_interface import ModelInterface, ExtractionResult
from typing import List, Dict
import numpy as np
import io
from PIL import Image
from loguru import logger
from src.utils.extraction_utils import intelligent_kvp_extraction # Import the new function

class PPStructureV2(ModelInterface):
    def __init__(self):
        self.ocr = None
        self.load_error = None
        try:
            logger.info("Initializing PaddleOCR (PP-StructureV2)...")
            from paddleocr import PaddleOCR
            # Initialize with standard English and Table support enabled
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            self.load_error = f"CRITICAL: Failed to load real PaddleOCR: {str(e)}"           
            logger.error(self.load_error)
            # We no longer switch to simulated mode. The service will report 
            # as degraded and fail extraction requests until the environment is fixed.
            raise
    
    async def extract(self, images: List[bytes], keys: Dict[str, List[str]]) -> ExtractionResult:
        if self.ocr is None:
            raise RuntimeError(self.load_error or "Model not initialized")
        """
        Extract key-value pairs from multiple images using PaddleOCR.
        Algorithmic Complexity: O(N * M) where N is number of pages and M is number of detected blocks.
        """
        all_kvps = {}
        all_confs = {}
        
        for i, img_bytes in enumerate(images):
            # Convert bytes back to numpy array for PaddleOCR
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            img_array = np.array(image)
            
            # Perform OCR
            result = self.ocr.ocr(img_array, cls=True)
            
            if not result or not result[0]:
                continue

            # Use the intelligent_kvp_extraction function
            extracted_data = intelligent_kvp_extraction(result[0], keys)
            
            # Populate all_kvps with extracted data
            for key, value in extracted_data.items():
                all_kvps[key] = value
                # For now, we'll assign a default confidence or re-evaluate how to get confidence
                # as intelligent_kvp_extraction doesn't return it yet.
                all_confs[key] = 0.9 # Placeholder confidence

        # Fill in missing keys with defaults
        for key in keys.keys(): # Iterate over the keys from the config
            if key not in all_kvps:
                all_kvps[key] = "Not Found"
                all_confs[key] = 0.0

        status = "approved" if all(c > 0.8 for c in all_confs.values()) else "review"
        return ExtractionResult(all_kvps, all_confs, status)