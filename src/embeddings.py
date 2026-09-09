import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Union
from src.model import CLIPModelWrapper
from src.utils import l2_normalize

def extract_feature_tensor(features) -> torch.Tensor:
    """Extracts the pooled 2D embedding tensor from PyTorch/transformers output."""
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
        return features.last_hidden_state[:, 0, :]
    return features[0]

class EmbeddingGenerator:
    """
    Handles image and text feature extraction using CLIP and applies L2 normalization.
    """

    def __init__(self, model_wrapper: Union[CLIPModelWrapper, None] = None):
        if model_wrapper is None:
            model_wrapper = CLIPModelWrapper.get_instance()
        self.model, self.processor = model_wrapper.get_model_and_processor()
        self.device = model_wrapper.device

    def generate_image_embeddings(
        self, image_paths: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Generates L2-normalized image embeddings in batches.

        Returns:
            Tuple of:
                - embeddings: np.ndarray of shape (N_success, embedding_dim)
                - successful_indices: List[int] of original indices that were
                  successfully processed, maintaining the guarantee that
                  embeddings[i] corresponds to image_paths[successful_indices[i]].
        """
        all_embeddings = []
        successful_indices: List[int] = []
        total = len(image_paths)

        print(f"Generating image embeddings for {total} images (batch size: {batch_size})...")

        for i in tqdm(range(0, total, batch_size), desc="Image Indexing"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            batch_successful_indices = []

            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    batch_successful_indices.append(i + j)
                except Exception as e:
                    print(f"\nWarning: Error loading image '{p}': {e}. Skipping.")

            if not images:
                continue

            inputs = self.processor(images=images, return_tensors="pt").to(self.device)

            with torch.no_grad():
                raw_features = self.model.get_image_features(**inputs)
                tensor_features = extract_feature_tensor(raw_features)

            # L2 Normalization in NumPy
            feats = tensor_features.cpu().numpy()
            norm_feats = l2_normalize(feats)
            all_embeddings.append(norm_feats)
            successful_indices.extend(batch_successful_indices)

        if not all_embeddings:
            return np.empty((0, 512), dtype=np.float32), []

        return np.vstack(all_embeddings).astype(np.float32), successful_indices

    def generate_text_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates L2-normalized text embeddings for single or multiple queries.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            raw_features = self.model.get_text_features(**inputs)
            tensor_features = extract_feature_tensor(raw_features)

        feats = tensor_features.cpu().numpy()
        return l2_normalize(feats).astype(np.float32)
