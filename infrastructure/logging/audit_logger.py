import logging, json
from datetime import datetime


class AuditLogger:

    def __init__(self):
        self.logger = logging.getLogger("aidr_audit")
        self.logger.setLevel(logging.INFO)

    def log(self, *, request_id: str, event: str, data: dict):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "event": event,
            "data": data
        }
        self.logger.info(json.dumps(record))
