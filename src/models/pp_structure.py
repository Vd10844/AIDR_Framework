from __future__ import annotations
from src.core.model_interface import ModelInterface, ExtractionResult
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import io
from PIL import Image
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# OCR Block Model (MUST BE DEFINED FIRST)
# -----------------------------
@dataclass
class OCRBlock:
    text: str
    confidence: float
    bbox: np.ndarray
    cx: float
    cy: float
    height: float
    order: int


# -----------------------------
# Field Configuration
# -----------------------------
FIELD_SPECS: Dict[str, Dict] = {
    "lot_number": {
        "aliases": ["lot no", "lot number", "homesite", "homesite #", "lot_no"],
        "required": True,
        "weight": 1.0,
        "value_pattern": r"^[A-Za-z0-9\-]+$",
    },
    "block": {
        "aliases": ["block", "blk"],
        "required": False,
        "weight": 0.7,
    },
    "address": {
        "aliases": ["addr", "site address", "property address", "location"],
        "required": True,
        "weight": 1.0,
    },
    "model_selected": {
        "aliases": ["model", "model type", "selected model", "house model"],
        "required": True,
        "weight": 0.9,
    },
    "elevation": {
        "aliases": ["elev", "front elevation", "design elevation"],
        "required": False,
        "weight": 0.8,
    },
    "garage_swing": {
        "aliases": ["garage", "garage orientation", "garage swing", "garage side"],
        "required": False,
        "weight": 0.7,
    },
    "external_structure": {
        "aliases": ["shed", "outbuilding", "external structure", "detached structure"],
        "required": False,
        "weight": 0.6,
    },
    "optional_notes": {
        "aliases": ["notes", "remarks", "comments", "optional notes"],
        "required": False,
        "weight": 0.5,
    },
}


# -----------------------------
# PPStructureV3 Implementation
# -----------------------------
class PPStructureV3(ModelInterface):
    def __init__(self, max_workers: int = 2):
        self.ocr = None
        self.load_error = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._field_cache = self._build_field_cache()

        try:
            logger.info("Initializing PaddleOCR (PPStructureV3)...")
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                det=True,
                rec=True,
                use_dilation=True,
                det_db_score_mode="slow",
                drop_score=0.3,
            )
            logger.info("PaddleOCR initialized successfully.")
        except Exception as e:
            self.load_error = f"CRITICAL: PaddleOCR init failed: {e}"
            logger.error(self.load_error)
            raise

    def _build_field_cache(self) -> Dict[str, List[str]]:
        """Cache normalized aliases for faster lookup"""
        cache = {}
        for field, spec in FIELD_SPECS.items():
            cache[field] = [alias.lower().replace(":", "").strip() 
                          for alias in spec["aliases"]]
        return cache

    # -----------------------------
    # Core API Implementation
    # -----------------------------
    async def extract(self, images: List[bytes], keys: List[str]) -> ExtractionResult:
        """Async implementation with proper error handling"""
        if not self.ocr:
            raise RuntimeError(self.load_error or "OCR not initialized")

        if not images:
            return self._create_empty_result(keys)

        try:
            # Process images in parallel
            tasks = [self._process_single_image(img_bytes, keys) 
                    for img_bytes in images]
            image_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            kvps, confidences = self._aggregate_results(image_results, keys)
            metadata = self._generate_metadata(image_results)
            
            return ExtractionResult(
                kvps=kvps,
                confidences=confidences,
                status=self._derive_status(confidences, keys),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return self._create_error_result(keys, str(e))

    async def _process_single_image(self, img_bytes: bytes, keys: List[str]) -> Dict:
        """Process single image in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._extract_from_image_sync,
            img_bytes,
            keys
        )

    def _extract_from_image_sync(self, img_bytes: bytes, keys: List[str]) -> Dict:
        """Synchronous extraction for thread pool"""
        blocks = self._extract_blocks(img_bytes)
        if not blocks:
            return {"kvps": {}, "confidences": {}, "blocks": []}

        doc_diag = self._document_diagonal(blocks)
        kvps = {}
        confidences = {}

        for key in keys:
            if key not in FIELD_SPECS:
                kvps[key] = "Not Found"
                confidences[key] = 0.0
                continue

            spec = FIELD_SPECS[key]
            key_block = self._find_key_block(blocks, spec["aliases"])
            
            if not key_block:
                kvps[key] = "Not Found"
                confidences[key] = 0.0
                continue

            value, score = self._resolve_value(key_block, blocks, doc_diag)
            if value:
                # Apply field-specific validation
                value = self._validate_field_value(key, value)
                kvps[key] = value
                confidences[key] = min(score * spec["weight"], 1.0)
            else:
                kvps[key] = "Not Found"
                confidences[key] = 0.0

        return {
            "kvps": kvps,
            "confidences": confidences,
            "blocks": [(b.text, b.confidence, (b.cx, b.cy)) for b in blocks]
        }

    def _validate_field_value(self, field: str, value: str) -> str:
        """Apply field-specific validation and cleaning"""
        spec = FIELD_SPECS.get(field, {})
        
        if "value_pattern" in spec:
            import re
            if re.match(spec["value_pattern"], value):
                return value.strip()
        
        # General cleaning
        value = value.strip()
        
        # Remove common OCR artifacts
        artifacts = ["|", "]", "[", "}", "{", "\\", "/", "\"", "'"]
        for artifact in artifacts:
            value = value.replace(artifact, "")
            
        return value

    def _aggregate_results(self, image_results: List, keys: List[str]) -> Tuple[Dict, Dict]:
        """Aggregate results from multiple images"""
        kvps = {key: "Not Found" for key in keys}
        confidences = {key: 0.0 for key in keys}
        
        for result in image_results:
            if isinstance(result, Exception):
                continue
                
            for key in keys:
                if key in result["kvps"]:
                    new_value = result["kvps"][key]
                    new_conf = result["confidences"][key]
                    
                    # Take highest confidence value
                    if new_conf > confidences[key] and new_value != "Not Found":
                        kvps[key] = new_value
                        confidences[key] = new_conf
        
        return kvps, confidences

    def _generate_metadata(self, image_results: List) -> Dict[str, Any]:
        """Generate metadata for the extraction result"""
        metadata = {
            "image_count": len(image_results),
            "successful_extractions": 0,
            "average_confidence": 0.0,
            "blocks_per_image": []
        }
        
        successful = 0
        total_confidence = 0.0
        conf_count = 0
        
        for result in image_results:
            if isinstance(result, Exception):
                continue
                
            successful += 1
            
            # Track blocks info
            if "blocks" in result:
                metadata["blocks_per_image"].append(len(result["blocks"]))
            
            # Accumulate confidences
            for conf in result.get("confidences", {}).values():
                if conf > 0:
                    total_confidence += conf
                    conf_count += 1
        
        if successful > 0:
            metadata["successful_extractions"] = successful
        if conf_count > 0:
            metadata["average_confidence"] = total_confidence / conf_count
        
        return metadata

    def _create_empty_result(self, keys: List[str]) -> ExtractionResult:
        """Create empty result for edge cases"""
        kvps = {key: "Not Found" for key in keys}
        confidences = {key: 0.0 for key in keys}
        return ExtractionResult(
            kvps=kvps,
            confidences=confidences,
            status="rejected",
            metadata={"error": "No images provided"}
        )

    def _create_error_result(self, keys: List[str], error_msg: str) -> ExtractionResult:
        """Create error result"""
        kvps = {key: "Not Found" for key in keys}
        confidences = {key: 0.0 for key in keys}
        return ExtractionResult(
            kvps=kvps,
            confidences=confidences,
            status="rejected",
            metadata={"error": error_msg}
        )

    def _derive_status(self, confidences: Dict[str, float], keys: List[str]) -> str:
        """Derive status with consideration for required fields"""
        if not confidences:
            return "rejected"
        
        # Check required fields
        required_fields = [k for k in keys if k in FIELD_SPECS and FIELD_SPECS[k].get("required", False)]
        missing_required = any(
            confidences.get(field, 0.0) < 0.5 for field in required_fields
        )
        
        if missing_required:
            return "rejected"
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for field in keys:
            if field in FIELD_SPECS:
                weight = FIELD_SPECS[field].get("weight", 1.0)
                conf = confidences.get(field, 0.0)
                weighted_sum += conf * weight
                total_weight += weight
        
        if total_weight == 0:
            return "rejected"
        
        avg = weighted_sum / total_weight
        
        if avg >= 0.85:
            return "approved"
        if avg >= 0.6:
            return "review"
        return "rejected"

    # -----------------------------
    # Optimized OCR Block Extraction
    # -----------------------------
    def _extract_blocks(self, img_bytes: bytes) -> List[OCRBlock]:
        """Extract OCR blocks with proper resource management"""
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Optional: preprocess image for better OCR
                img = self._preprocess_image(img)
                
                img_array = np.array(img)
            
            # Perform OCR
            ocr_result = self.ocr.ocr(img_array, cls=True)
            
            if not ocr_result or not ocr_result[0]:
                return []
            
            blocks: List[OCRBlock] = []
            
            for idx, line in enumerate(ocr_result[0]):
                text, conf = line[1]
                bbox = np.array(line[0])
                
                # Filter low confidence results
                if conf < 0.3:
                    continue
                
                cy_vals = bbox[:, 1]
                height = cy_vals.max() - cy_vals.min()
                
                # Filter very small text (likely noise)
                if height < 5:
                    continue
                
                blocks.append(
                    OCRBlock(
                        text=text.strip(),
                        confidence=self._clamp_conf(conf),
                        bbox=bbox,
                        cx=bbox[:, 0].mean(),
                        cy=bbox[:, 1].mean(),
                        height=height,
                        order=idx,
                    )
                )
            
            return blocks
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Simple image preprocessing for better OCR results"""
        # Convert to grayscale for better contrast
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Resize if too small (maintain aspect ratio)
        min_dpi = 150
        current_dpi = image.info.get('dpi', (72, 72))[0]
        
        if current_dpi < min_dpi:
            scale = min_dpi / current_dpi
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image

    # -----------------------------
    # Optimized Key Matching
    # -----------------------------
    def _find_key_block(self, blocks: List[OCRBlock], aliases: List[str]) -> Optional[OCRBlock]:
        """Fast key block matching using cached aliases"""
        # Normalize aliases once
        normalized_aliases = [alias.lower().replace(":", "").strip() for alias in aliases]
        
        for block in blocks:
            text = block.text.lower()
            
            # Quick checks
            if ":" in text:
                text = text.split(":")[0].strip()
            
            # Direct matching
            if text in normalized_aliases:
                return block
            
            # Partial matching
            for alias in normalized_aliases:
                if alias in text:
                    return block
        
        return None

    # -----------------------------
    # Optimized Value Resolution
    # -----------------------------
    def _resolve_value(
        self,
        key_block: OCRBlock,
        blocks: List[OCRBlock],
        doc_diag: float,
    ) -> Tuple[Optional[str], float]:
        
        # Create spatial index for faster lookup
        spatial_index = self._build_spatial_index(blocks)
        
        # Find candidates in proximity
        candidates = self._find_nearby_blocks(key_block, blocks, spatial_index)
        
        if not candidates:
            return None, 0.0
        
        # Score candidates
        scored = []
        for candidate in candidates:
            prior = self._positional_prior(key_block, candidate)
            if prior == 0.0:
                continue
            
            score = self._score_candidate(key_block, candidate, prior, doc_diag)
            scored.append((score, candidate))
        
        if not scored:
            return None, 0.0
        
        # Select best candidate
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_block = scored[0]
        
        return best_block.text, best_score

    def _build_spatial_index(self, blocks: List[OCRBlock]) -> Optional[Any]:
        """Build spatial index for faster proximity search"""
        try:
            from scipy.spatial import KDTree
            points = np.array([[b.cx, b.cy] for b in blocks])
            if len(points) > 0:
                return KDTree(points)
            return None
        except ImportError:
            logger.warning("scipy not installed, using linear search")
            return None

    def _find_nearby_blocks(self, key_block: OCRBlock, blocks: List[OCRBlock], 
                           kdtree: Optional[Any], radius: float = 200.0) -> List[OCRBlock]:
        """Find blocks within a radius using spatial index"""
        if kdtree is None:
            # Fallback to linear search
            nearby_blocks = []
            for block in blocks:
                if block is key_block:
                    continue
                dist = np.hypot(block.cx - key_block.cx, block.cy - key_block.cy)
                if dist < radius:
                    nearby_blocks.append(block)
            return nearby_blocks
        
        try:
            from scipy.spatial import KDTree
            point = np.array([[key_block.cx, key_block.cy]])
            indices = kdtree.query_ball_point(point[0], r=radius)
            
            nearby_blocks = []
            for idx in indices:
                block = blocks[idx]
                if block is not key_block:
                    nearby_blocks.append(block)
            
            return nearby_blocks
        except Exception as e:
            logger.warning(f"KDTree search failed: {e}, falling back to linear search")
            return []

    # -----------------------------
    # Positional Prior (Layout Logic)
    # -----------------------------
    def _positional_prior(self, key: OCRBlock, cand: OCRBlock) -> float:
        row_tol = max(key.height, cand.height) * 0.6
        col_tol = row_tol

        # Same row → right
        if abs(cand.cy - key.cy) < row_tol and cand.cx > key.cx:
            return 1.0

        # Same column → below
        if abs(cand.cx - key.cx) < col_tol and cand.cy > key.cy:
            return 0.8

        # Fallback proximity
        return 0.3

    # -----------------------------
    # Scoring Function
    # -----------------------------
    def _score_candidate(
        self,
        key: OCRBlock,
        cand: OCRBlock,
        prior: float,
        doc_diag: float,
    ) -> float:
        dist = np.hypot(cand.cx - key.cx, cand.cy - key.cy)
        norm_dist = dist / max(doc_diag, 1.0)
        return prior * cand.confidence / (1.0 + norm_dist)

    # -----------------------------
    # Utilities
    # -----------------------------
    def _document_diagonal(self, blocks: List[OCRBlock]) -> float:
        xs = [b.cx for b in blocks]
        ys = [b.cy for b in blocks]
        return np.hypot(max(xs) - min(xs), max(ys) - min(ys))

    def _clamp_conf(self, conf: float) -> float:
        return max(0.5, min(conf, 0.98))

    # -----------------------------
    # Cleanup
    # -----------------------------
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        self.ocr = None