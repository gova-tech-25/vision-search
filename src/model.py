import torch
from transformers import CLIPModel, CLIPProcessor
from typing import Tuple, Optional
from src.config import MODEL_NAME, DEVICE

class CLIPModelWrapper:
    """
    Singleton wrapper for pretrained CLIP model and processor.
    """
    _instance: Optional["CLIPModelWrapper"] = None

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        print(f"Loading CLIP model '{self.model_name}' on device: '{self.device}'...")

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        print("CLIP model and processor loaded successfully.")

    @classmethod
    def get_instance(cls, model_name: str = MODEL_NAME, device: str = DEVICE) -> "CLIPModelWrapper":
        if cls._instance is None:
            cls._instance = cls(model_name, device)
        return cls._instance

    def get_model_and_processor(self) -> Tuple[CLIPModel, CLIPProcessor]:
        return self.model, self.processor
