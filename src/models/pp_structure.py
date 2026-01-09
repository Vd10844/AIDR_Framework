from __future__ import annotations
from src.core.model_interface import ModelInterface, ExtractionResult
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
import numpy as np
import io
import re
from PIL import Image, ImageEnhance
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from enum import Enum


class LayoutType(Enum):
    """Types of document layouts"""
    TABULAR = "tabular"
    KEY_VALUE_PAIRS = "key_value_pairs"
    FORM = "form"
    REPORT = "report"
    UNSTRUCTURED = "unstructured"


@dataclass
class OCRBlock:
    text: str
    confidence: float
    bbox: np.ndarray  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    cx: float  # Center x
    cy: float  # Center y
    height: float
    width: float
    order: int
    line_group: int = -1
    column_group: int = -1
    is_key_candidate: bool = False
    is_value_candidate: bool = False


# -----------------------------
# Field Configuration with Multiple Extraction Strategies
# -----------------------------
FIELD_SPECS: Dict[str, Dict] = {
    "lot_number": {
        "aliases": ["lot no", "lot number", "homesite", "homesite #", "lot_no", "lot id", "lot"],
        "required": True,
        "weight": 1.0,
        "regex_patterns": [
            r"lot\s*(?:no|number|id)?[:\s]*([A-Z0-9\-]+)",
            r"homesite\s*[#]?\s*(\d+)",
            r"lot[:\s]*(\d+)"
        ],
        "context_words": ["lot", "homesite", "site"],
        "value_type": "alphanumeric",
    },
    "block": {
        "aliases": ["block", "blk"],
        "required": False,
        "weight": 0.7,
        "regex_patterns": [r"block\s*(?:no|number)?[:\s]*([A-Z0-9]+)", r"blk[:\s]*(\w+)"],
        "context_words": ["block", "section"],
        "value_type": "alphanumeric",
    },
    "address": {
        "aliases": ["addr", "site address", "property address", "location", "address"],
        "required": True,
        "weight": 1.0,
        "regex_patterns": [r"address[:\s]*(.+?)(?:\n|$)", r"location[:\s]*(.+?)(?:\n|$)"],
        "context_words": ["address", "location", "street"],
        "value_type": "text",
    },
    "model_selected": {
        "aliases": ["model", "model type", "selected model", "house model", "model"],
        "required": True,
        "weight": 0.9,
        "regex_patterns": [r"model[:\s]*(.+?)(?:\n|$)", r"type[:\s]*(.+?)(?:\n|$)"],
        "context_words": ["model", "type", "design"],
        "value_type": "text",
    },
    "elevation": {
        "aliases": ["elev", "front elevation", "design elevation", "elevation"],
        "required": False,
        "weight": 0.8,
        "regex_patterns": [r"elevation[:\s]*(.+?)(?:\n|$)", r"elev[:\s]*(.+?)(?:\n|$)"],
        "context_words": ["elevation", "design", "front"],
        "value_type": "text",
    },
    "garage_swing": {
        "aliases": ["garage", "garage orientation", "garage swing", "garage side", "swing"],
        "required": False,
        "weight": 0.7,
        "regex_patterns": [
            r"garage\s*swing[:\s\-]*([A-Z]+)",
            r"garage\s*orientation[:\s]*(.+?)(?:\n|$)",
            r"swing[:\s]*([A-Z]+)"
        ],
        "context_words": ["garage", "swing", "orientation", "side"],
        "value_type": "directional",
    },
    "external_structure": {
        "aliases": ["shed", "outbuilding", "external structure", "detached structure"],
        "required": False,
        "weight": 0.6,
        "regex_patterns": [r"shed[:\s]*(.+?)(?:\n|$)", r"outbuilding[:\s]*(.+?)(?:\n|$)"],
        "context_words": ["shed", "structure", "outbuilding"],
        "value_type": "text",
    },
    "optional_notes": {
        "aliases": ["notes", "remarks", "comments", "optional notes"],
        "required": False,
        "weight": 0.5,
        "regex_patterns": [r"notes[:\s]*(.+?)(?:\n|$)", r"comments[:\s]*(.+?)(?:\n|$)"],
        "context_words": ["note", "comment", "remark"],
        "value_type": "text",
    },
}


# -----------------------------
# PPStructureV3 with Adaptive Layout Detection
# -----------------------------
class PPStructureV3(ModelInterface):
    def __init__(self, max_workers: int = 2):
        self.ocr = None
        self.load_error = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._field_cache = self._build_field_cache()
        self._regex_cache = self._build_regex_cache()
        
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
                show_log=False,
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
            cache[field] = [alias.lower().strip() for alias in spec["aliases"]]
        return cache

    def _build_regex_cache(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for faster matching"""
        cache = {}
        for field, spec in FIELD_SPECS.items():
            patterns = spec.get("regex_patterns", [])
            cache[field] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return cache

    # -----------------------------
    # Core Extraction with Multiple Strategies
    # -----------------------------
    async def extract(self, images: List[bytes], keys: List[str]) -> ExtractionResult:
        """Adaptive extraction using multiple strategies"""
        if not self.ocr:
            raise RuntimeError(self.load_error or "OCR not initialized")

        if not images:
            return self._create_empty_result(keys)

        try:
            # Process all images
            all_blocks = []
            all_texts = []
            
            for img_bytes in images:
                blocks = self._extract_blocks(img_bytes)
                if blocks:
                    all_blocks.extend(blocks)
                    # Extract text in reading order
                    sorted_blocks = sorted(blocks, key=lambda b: (b.cy, b.cx))
                    text = " ".join([b.text for b in sorted_blocks])
                    all_texts.append(text)
            
            if not all_blocks:
                return self._create_empty_result(keys)
            
            # Analyze document layout
            layout_type = self._detect_layout(all_blocks)
            logger.info(f"Detected layout type: {layout_type}")
            
            # Apply appropriate extraction strategy
            kvps = {}
            confidences = {}
            
            for key in keys:
                if key not in FIELD_SPECS:
                    kvps[key] = "Not Found"
                    confidences[key] = 0.0
                    continue
                
                spec = FIELD_SPECS[key]
                extraction_results = []
                
                # Strategy 1: Regex-based extraction from full text
                regex_result = self._extract_by_regex(key, "\n".join(all_texts))
                if regex_result:
                    extraction_results.append(("regex", regex_result, spec["weight"]))
                
                # Strategy 2: Layout-aware extraction
                if layout_type == LayoutType.TABULAR:
                    layout_result = self._extract_from_tabular(key, all_blocks, spec)
                elif layout_type == LayoutType.REPORT:
                    layout_result = self._extract_from_report(key, all_blocks, spec)
                else:
                    layout_result = self._extract_from_unstructured(key, all_blocks, spec)
                
                if layout_result:
                    extraction_results.append(("layout", layout_result, spec["weight"] * 0.9))
                
                # Strategy 3: Proximity-based extraction
                proximity_result = self._extract_by_proximity(key, all_blocks, spec)
                if proximity_result:
                    extraction_results.append(("proximity", proximity_result, spec["weight"] * 0.8))
                
                # Choose the best result
                if extraction_results:
                    # Sort by confidence
                    extraction_results.sort(key=lambda x: x[2], reverse=True)
                    best_strategy, best_value, best_confidence = extraction_results[0]
                    
                    # Validate the extracted value
                    validated_value = self._validate_value(key, best_value)
                    
                    kvps[key] = validated_value
                    confidences[key] = min(best_confidence, 1.0)
                    logger.debug(f"Extracted {key}={validated_value} using {best_strategy}")
                else:
                    kvps[key] = "Not Found"
                    confidences[key] = 0.0
            
            # Ensure all requested keys exist
            for key in keys:
                kvps.setdefault(key, "Not Found")
                confidences.setdefault(key, 0.0)
            
            metadata = {
                "layout_type": layout_type.value,
                "total_blocks": len(all_blocks),
                "extraction_strategies": len(extraction_results) if extraction_results else 0,
            }
            
            status = self._derive_status(confidences, keys)
            return ExtractionResult(kvps, confidences, status, metadata)
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return self._create_error_result(keys, str(e))

    # -----------------------------
    # Layout Detection
    # -----------------------------
    def _detect_layout(self, blocks: List[OCRBlock]) -> LayoutType:
        """Detect the type of document layout"""
        if len(blocks) < 5:
            return LayoutType.UNSTRUCTURED
        
        # Check for tabular structure
        if self._has_tabular_structure(blocks):
            return LayoutType.TABULAR
        
        # Check for report structure (headers, sections)
        if self._has_report_structure(blocks):
            return LayoutType.REPORT
        
        # Check for form-like structure (labels and inputs)
        if self._has_form_structure(blocks):
            return LayoutType.FORM
        
        # Check for key-value pairs
        if self._has_key_value_structure(blocks):
            return LayoutType.KEY_VALUE_PAIRS
        
        return LayoutType.UNSTRUCTURED

    def _has_tabular_structure(self, blocks: List[OCRBlock]) -> bool:
        """Check if blocks are arranged in a grid/table"""
        if len(blocks) < 10:
            return False
        
        # Group by approximate rows
        rows = defaultdict(list)
        for block in blocks:
            row_key = round(block.cy / 10) * 10  # Group by vertical position
            rows[row_key].append(block)
        
        # Check if we have multiple rows with multiple columns
        row_counts = [len(rows[key]) for key in rows]
        avg_columns = sum(row_counts) / len(row_counts)
        
        # A table should have at least 3 rows and average > 2 columns
        return len(rows) >= 3 and avg_columns >= 2

    def _has_report_structure(self, blocks: List[OCRBlock]) -> bool:
        """Check if document has report-like structure"""
        # Look for report indicators
        report_indicators = ["report", "table", "summary", "detail", "total", "page"]
        block_texts = " ".join([b.text.lower() for b in blocks])
        
        indicator_count = sum(1 for indicator in report_indicators if indicator in block_texts)
        return indicator_count >= 2

    def _has_form_structure(self, blocks: List[OCRBlock]) -> bool:
        """Check if document has form-like structure"""
        # Look for form indicators (labels with colons, checkboxes, etc.)
        form_indicators = [":", "☐", "□", "✔", "yes", "no", "date", "signature"]
        indicator_count = 0
        
        for block in blocks:
            text = block.text.lower()
            if any(indicator in text for indicator in form_indicators):
                indicator_count += 1
        
        return indicator_count >= len(blocks) * 0.2  # At least 20% of blocks are form-like

    def _has_key_value_structure(self, blocks: List[OCRBlock]) -> bool:
        """Check for colon-separated key-value pairs"""
        colon_count = sum(1 for block in blocks if ":" in block.text)
        return colon_count >= max(3, len(blocks) * 0.1)  # At least 3 or 10% of blocks

    # -----------------------------
    # Extraction Strategies
    # -----------------------------
    def _extract_by_regex(self, field: str, text: str) -> Optional[str]:
        """Extract using regex patterns"""
        patterns = self._regex_cache.get(field, [])
        
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                # Return the first non-empty match
                for match in matches:
                    if isinstance(match, tuple):
                        # If pattern has groups, use the first non-empty group
                        for group in match:
                            if group and str(group).strip():
                                return str(group).strip()
                    elif match and str(match).strip():
                        return str(match).strip()
        
        return None

    def _extract_from_tabular(self, field: str, blocks: List[OCRBlock], spec: Dict) -> Optional[str]:
        """Extract from tabular data by finding column headers"""
        # Group blocks into rows
        rows = defaultdict(list)
        for block in blocks:
            row_key = round(block.cy / 10) * 10
            rows[row_key].append(block)
        
        # Sort rows by Y position
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])
        
        # Find header row (usually first row or row with field aliases)
        header_row_idx = -1
        for idx, (row_key, row_blocks) in enumerate(sorted_rows):
            row_text = " ".join([b.text.lower() for b in row_blocks])
            if any(alias in row_text for alias in spec["aliases"]):
                header_row_idx = idx
                break
        
        if header_row_idx == -1:
            # Try to find any row that might be a header
            for idx, (row_key, row_blocks) in enumerate(sorted_rows):
                if idx == 0:  # First row is often header
                    header_row_idx = idx
                    break
        
        if header_row_idx == -1 or header_row_idx >= len(sorted_rows) - 1:
            return None
        
        # Get the column index of our field
        header_blocks = sorted_rows[header_row_idx][1]
        target_col_idx = -1
        
        for idx, block in enumerate(header_blocks):
            block_text = block.text.lower()
            if any(alias in block_text for alias in spec["aliases"]):
                target_col_idx = idx
                break
        
        if target_col_idx == -1:
            return None
        
        # Look for values in rows below the header
        for row_idx in range(header_row_idx + 1, len(sorted_rows)):
            row_blocks = sorted_rows[row_idx][1]
            if target_col_idx < len(row_blocks):
                value_block = row_blocks[target_col_idx]
                if value_block.text.strip():
                    return value_block.text.strip()
        
        return None

    def _extract_from_report(self, field: str, blocks: List[OCRBlock], spec: Dict) -> Optional[str]:
        """Extract from report-style documents"""
        # Sort blocks by reading order
        sorted_blocks = sorted(blocks, key=lambda b: (b.cy, b.cx))
        
        # Look for patterns like "Field: value" or "Field is value"
        for i, block in enumerate(sorted_blocks):
            block_text = block.text.lower()
            
            # Check if this block contains a field alias
            if any(alias in block_text for alias in spec["aliases"]):
                # Check current block for value after colon/dash
                if ":" in block.text:
                    parts = block.text.split(":", 1)
                    if len(parts) == 2 and parts[1].strip():
                        return parts[1].strip()
                
                if "-" in block.text:
                    parts = block.text.split("-", 1)
                    if len(parts) == 2 and parts[1].strip():
                        return parts[1].strip()
                
                # Check next block (common in reports)
                if i + 1 < len(sorted_blocks):
                    next_block = sorted_blocks[i + 1]
                    # Check if next block is to the right or below
                    if (abs(next_block.cy - block.cy) < block.height * 2 or
                        next_block.cx > block.cx):
                        if next_block.text.strip():
                            return next_block.text.strip()
        
        return None

    def _extract_from_unstructured(self, field: str, blocks: List[OCRBlock], spec: Dict) -> Optional[str]:
        """Extract from unstructured text using proximity and context"""
        # Build spatial index
        from scipy.spatial import KDTree
        points = np.array([[b.cx, b.cy] for b in blocks])
        kdtree = KDTree(points)
        
        # Find blocks that might be keys
        key_blocks = []
        for i, block in enumerate(blocks):
            block_text = block.text.lower()
            if any(alias in block_text for alias in spec["aliases"]):
                key_blocks.append((i, block))
        
        if not key_blocks:
            return None
        
        # For each key block, look for nearby value blocks
        best_value = None
        best_score = 0
        
        for key_idx, key_block in key_blocks:
            # Find blocks within search radius
            search_radius = key_block.height * 10  # Adaptive radius
            nearby_indices = kdtree.query_ball_point([key_block.cx, key_block.cy], search_radius)
            
            for idx in nearby_indices:
                if idx == key_idx:
                    continue
                
                value_block = blocks[idx]
                
                # Calculate positional score
                score = self._calculate_position_score(key_block, value_block)
                
                # Boost score if block looks like a value (not another key)
                if not self._looks_like_key(value_block.text, spec):
                    score *= 1.2
                
                if score > best_score and value_block.text.strip():
                    best_score = score
                    best_value = value_block.text.strip()
        
        return best_value

    def _extract_by_proximity(self, field: str, blocks: List[OCRBlock], spec: Dict) -> Optional[str]:
        """Simple proximity-based extraction"""
        # Find key blocks
        key_blocks = []
        for block in blocks:
            block_text = block.text.lower()
            if any(alias in block_text for alias in spec["aliases"]):
                key_blocks.append(block)
        
        if not key_blocks:
            return None
        
        # For each key block, find the closest non-key block
        best_value = None
        min_distance = float('inf')
        
        for key_block in key_blocks:
            for value_block in blocks:
                if value_block is key_block:
                    continue
                
                # Skip if value block looks like a key
                if self._looks_like_key(value_block.text, spec):
                    continue
                
                distance = np.hypot(value_block.cx - key_block.cx, 
                                   value_block.cy - key_block.cy)
                
                if distance < min_distance and value_block.text.strip():
                    min_distance = distance
                    best_value = value_block.text.strip()
        
        return best_value

    # -----------------------------
    # Helper Methods
    # -----------------------------
    def _calculate_position_score(self, key_block: OCRBlock, value_block: OCRBlock) -> float:
        """Calculate how likely value_block is the value for key_block"""
        score = 1.0
        
        # Same row, to the right (best case)
        if abs(value_block.cy - key_block.cy) < key_block.height * 0.5 and value_block.cx > key_block.cx:
            score *= 2.0
        
        # Below, roughly aligned
        elif value_block.cy > key_block.cy and abs(value_block.cx - key_block.cx) < key_block.width * 2:
            score *= 1.5
        
        # Adjust by distance
        distance = np.hypot(value_block.cx - key_block.cx, value_block.cy - key_block.cy)
        score /= (1 + distance / 100)  # Normalize
        
        # Adjust by confidence
        score *= value_block.confidence
        
        return score

    def _looks_like_key(self, text: str, spec: Dict) -> bool:
        """Check if text looks like a key rather than a value"""
        text_lower = text.lower()
        
        # Check if it contains field aliases
        if any(alias in text_lower for alias in spec["aliases"]):
            return True
        
        # Check if it ends with colon (common for keys)
        if text.strip().endswith(":"):
            return True
        
        # Check if it's very short (more likely to be a key)
        words = text.split()
        if len(words) <= 2 and len(text) < 20:
            return True
        
        return False

    def _validate_value(self, field: str, value: str) -> str:
        """Validate and clean extracted value"""
        if not value:
            return "Not Found"
        
        value = value.strip()
        spec = FIELD_SPECS.get(field, {})
        value_type = spec.get("value_type", "text")
        
        # Type-specific cleaning
        if value_type == "alphanumeric":
            # Remove non-alphanumeric except hyphens
            value = re.sub(r'[^A-Za-z0-9\-]', '', value)
        elif value_type == "directional":
            # Extract direction (LEFT, RIGHT, etc.)
            direction_match = re.search(r'(LEFT|RIGHT|L|R)', value, re.IGNORECASE)
            if direction_match:
                value = direction_match.group(1).upper()
        
        # Remove common OCR artifacts
        artifacts = ["|", "]", "[", "}", "{", "\\", "/", "\"", "'", "`", "*"]
        for artifact in artifacts:
            value = value.replace(artifact, "")
        
        # Remove extra whitespace
        value = re.sub(r'\s+', ' ', value).strip()
        
        return value if value else "Not Found"

    def _extract_blocks(self, img_bytes: bytes) -> List[OCRBlock]:
        """Extract OCR blocks from image"""
        try:
            with Image.open(io.BytesIO(img_bytes)) as img:
                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Basic preprocessing
                img = self._preprocess_image(img)
                img_array = np.array(img)
            
            # Perform OCR
            ocr_result = self.ocr.ocr(img_array, cls=True)
            
            if not ocr_result or not ocr_result[0]:
                return []
            
            blocks = []
            for idx, line in enumerate(ocr_result[0]):
                text, conf = line[1]
                bbox = np.array(line[0])
                
                # Filter low confidence
                if conf < 0.3:
                    continue
                
                # Calculate block properties
                x_coords = bbox[:, 0]
                y_coords = bbox[:, 1]
                width = x_coords.max() - x_coords.min()
                height = y_coords.max() - y_coords.min()
                
                blocks.append(
                    OCRBlock(
                        text=text.strip(),
                        confidence=min(max(conf, 0.3), 0.98),
                        bbox=bbox,
                        cx=np.mean(x_coords),
                        cy=np.mean(y_coords),
                        height=height,
                        width=width,
                        order=idx,
                    )
                )
            
            return blocks
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Simple image preprocessing"""
        # Convert to grayscale for better OCR
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        
        return image

    def _create_empty_result(self, keys: List[str]) -> ExtractionResult:
        kvps = {key: "Not Found" for key in keys}
        confidences = {key: 0.0 for key in keys}
        return ExtractionResult(
            kvps=kvps,
            confidences=confidences,
            status="rejected",
            metadata={"error": "No images provided"}
        )

    def _create_error_result(self, keys: List[str], error_msg: str) -> ExtractionResult:
        kvps = {key: "Not Found" for key in keys}
        confidences = {key: 0.0 for key in keys}
        return ExtractionResult(
            kvps=kvps,
            confidences=confidences,
            status="rejected",
            metadata={"error": error_msg}
        )

    def _derive_status(self, confidences: Dict[str, float], keys: List[str]) -> str:
        """Derive overall extraction status"""
        if not confidences:
            return "rejected"
        
        # Check required fields
        required_fields = [
            k for k in keys 
            if k in FIELD_SPECS and FIELD_SPECS[k].get("required", False)
        ]
        
        missing_required = any(
            confidences.get(field, 0.0) < 0.3 for field in required_fields
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
        
        if avg >= 0.7:
            return "approved"
        elif avg >= 0.4:
            return "review"
        else:
            return "rejected"

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)