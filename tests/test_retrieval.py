import pytest
import numpy as np
import faiss
from src.utils import l2_normalize, verify_l2_norm

def test_cosine_similarity_ordering():
    # Synthetic target vector
    query_vec = l2_normalize(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    # Synthetic image vectors
    # img1: identical direction (sim ~ 1.0)
    # img2: orthogonal direction (sim ~ 0.0)
    # img3: opposite direction (sim ~ -1.0)
    # img4: partial similarity (sim ~ 0.707)
    raw_images = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    image_vecs = l2_normalize(raw_images)

    # Compute dot products
    sims = np.dot(image_vecs, query_vec.T).squeeze(-1)
    top_indices = np.argsort(-sims)

    # Highest similarity should be index 0 (img1), followed by index 3 (img4)
    assert top_indices[0] == 0
    assert top_indices[1] == 3
    assert top_indices[2] == 1
    assert top_indices[3] == 2

    assert sims[top_indices[0]] > sims[top_indices[1]] > sims[top_indices[2]] > sims[top_indices[3]]

def test_faiss_flat_and_hnsw_equivalence():
    """Verify that FAISS FlatIP and HNSW return consistent top results on normalized vectors."""
    np.random.seed(42)
    dim = 512
    num_vectors = 500

    # Create random unit vectors
    raw = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = l2_normalize(raw)
    assert verify_l2_norm(vectors)

    # Build FlatIP index
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(vectors)

    # Build HNSW index
    hnsw_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.hnsw.efSearch = 64
    hnsw_index.add(vectors)

    # Query with a random unit vector
    query = l2_normalize(np.random.randn(1, dim).astype(np.float32))

    k = 10
    flat_scores, flat_indices = flat_index.search(query, k)
    hnsw_scores, hnsw_indices = hnsw_index.search(query, k)

    # Top-1 should match on small-to-medium dataset with high efSearch
    assert flat_indices[0, 0] == hnsw_indices[0, 0]
    # Recall@10 between HNSW and FlatIP exact search
    overlap = len(set(flat_indices[0]).intersection(set(hnsw_indices[0])))
    assert overlap >= 9  # HNSW should have >= 90% overlap with exact FlatIP
