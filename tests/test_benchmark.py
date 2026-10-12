import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import faiss
from src.utils import l2_normalize, verify_l2_norm


def test_hnsw_efsearch_recall_sweep():
    """
    Test that increasing efSearch monotonically non-decreases or maintains
    high recall against exact FlatIP ground truth on synthetic embeddings.
    """
    np.random.seed(42)
    dim = 512
    num_vectors = 1000
    k = 10

    # Synthetic dataset
    raw = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = l2_normalize(raw)

    # FlatIP (Exact)
    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)

    # HNSW (Approximate)
    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(vectors)

    # Queries
    num_queries = 20
    query_vectors = l2_normalize(np.random.randn(num_queries, dim).astype(np.float32))

    # Ground truth
    _, exact_indices = flat.search(query_vectors, k)

    recalls = []
    ef_values = [16, 32, 64, 128]

    for ef in ef_values:
        hnsw.hnsw.efSearch = ef
        _, approx_indices = hnsw.search(query_vectors, k)

        query_recalls = []
        for q in range(num_queries):
            exact_set = set(exact_indices[q])
            approx_set = set(approx_indices[q])
            overlap = len(exact_set & approx_set)
            query_recalls.append(overlap / k)

        recalls.append(float(np.mean(query_recalls)))

    # Basic validations
    assert recalls[0] >= 0.70  # Even lowest efSearch achieves >= 70%
    assert recalls[-1] >= recalls[0]  # Higher efSearch improves or equals lower
    assert recalls[-1] >= 0.90  # High efSearch achieves >= 90% recall
    print(f"HNSW parameter sweep verified: recalls={recalls}")


if __name__ == "__main__":
    test_hnsw_efsearch_recall_sweep()
    print("ALL TESTS PASSED")

