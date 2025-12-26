import yaml
from src.core.model_interface import ModelInterface, ExtractionResult

class DocumentPipeline:
    def __init__(self, config_path: str):
        print(f"Loading config: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.model = self._create_model()
    
    def _create_model(self):
        model_name = self.config['model']['name']
        print(f"Creating model: {model_name}")
        if model_name == "PP-StructureV2":
            from src.models.pp_structure import PPStructureV2
            return PPStructureV2()
        raise ValueError(f"Model {model_name} not supported")
    
    async def extract(self, file_bytes):
        images = [b"demo_image"]  # Demo mode
        result = await self.model.extract(images, self.config['model']['keys'])
        return result
