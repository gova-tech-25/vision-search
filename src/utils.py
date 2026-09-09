import time
import numpy as np
from typing import Dict, Any

class Timer:
    """Context manager for timing execution code blocks in milliseconds."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval_ms = (self.end - self.start) * 1000.0

def verify_l2_norm(embeddings: np.ndarray, atol: float = 1e-3) -> bool:
    """
    Verifies that rows of an embedding matrix are unit L2-normalized.
    """
    norms = np.linalg.norm(embeddings, axis=-1)
    return np.allclose(norms, 1.0, atol=atol)

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    Applies L2 normalization along the last axis.
    """
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms
