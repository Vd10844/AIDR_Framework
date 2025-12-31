from pathlib import Path
import shutil
from datetime import datetime

class HITLStorage:
    def __init__(self, base_path: str = "data/hitl"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_for_review(self, file_name: str, content: bytes, request_id: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = self.base_path / f"{timestamp}_{request_id}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = target_dir / file_name
        with open(file_path, "wb") as f:
            f.write(content)
        
        return str(file_path)

hitl_storage = HITLStorage()
