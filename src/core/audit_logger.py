import sys
from loguru import logger
import json
from datetime import datetime
from pathlib import Path

class AuditLogger:
    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure loguru to write to a file with JSON format
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level="INFO")
        logger.add(
            self.log_path, 
            rotation="10 MB", 
            serialize=True, 
            level="INFO",
            format="{time} {level} {message}"
        )

    def log_extraction(self, request_id: str, file_name: str, result: dict, status: str):
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "file_name": file_name,
            "status": status,
            "result_summary": {
                "keys_extracted": list(result.get("kvps", {}).keys()),
                "avg_confidence": sum(result.get("confidences", {}).values()) / len(result.get("confidences", {})) if result.get("confidences") else 0
            }
        }
        logger.info(f"AUDIT | {json.dumps(audit_entry)}")

audit_logger = AuditLogger()
