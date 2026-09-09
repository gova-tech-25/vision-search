import pytest
import numpy as np
from src.embeddings import EmbeddingGenerator
from src.utils import l2_normalize, verify_l2_norm

def test_l2_normalize_utility():
    raw = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(raw)

    # Check non-zero vector norm equals 1.0
    assert np.isclose(np.linalg.norm(normalized[0]), 1.0)
    assert np.isclose(normalized[0, 0], 0.6)
    assert np.isclose(normalized[0, 1], 0.8)

def test_text_embedding_generation_and_norm():
    generator = EmbeddingGenerator()
    texts = ["A dog playing in the grass", "A person riding a bicycle"]
    embeddings = generator.generate_text_embeddings(texts)

    # Assert shape (2, 512)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 512

    # Assert unit norm
    assert verify_l2_norm(embeddings, atol=1e-3)
