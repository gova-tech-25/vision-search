import json
import time
import numpy as np
import faiss

from src.config import (
    EMBEDDINGS_PATH,
    METADATA_PATH,
    FAISS_INDEX_PATH,
    HNSW_INDEX_PATH,
    HNSW_EF_SEARCH,
)

print("========================================")
print("WEEK 2 - HNSW RECALL BENCHMARK")
print("========================================")

# Load existing artifacts
embeddings = np.load(EMBEDDINGS_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

flat = faiss.read_index(str(FAISS_INDEX_PATH))
hnsw = faiss.read_index(str(HNSW_INDEX_PATH))

print(f"Images: {len(metadata)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"FlatIP vectors: {flat.ntotal}")
print(f"HNSW vectors: {hnsw.ntotal}")

# --------------------------------------------------
# Test queries
# --------------------------------------------------

queries = [
    "three women sitting in a village",
    "a dog running through grass",
    "children playing outside",
    "a man riding a bicycle",
    "a person skiing on snow",
    "people walking on a beach",
    "a group of people eating together",
    "a woman holding a baby",
    "a black dog playing with a ball",
    "a person standing near a building",
]

# --------------------------------------------------
# IMPORTANT:
# We use precomputed text embeddings only through
# the CLIP generator.
# --------------------------------------------------

from src.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()

K = 5

ef_values = [16, 32, 64, 128, 256]

print("\n========================================")
print("Generating query embeddings")
print("========================================")

query_embeddings = []

for query in queries:
    embedding = generator.generate_text_embeddings(query)
    query_embeddings.append(embedding[0])

query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

print(f"Query embeddings shape: {query_embeddings.shape}")

# --------------------------------------------------
# FlatIP ground truth
# --------------------------------------------------

print("\n========================================")
print("Computing exact FlatIP ground truth")
print("========================================")

flat_results = {}

for query, vector in zip(queries, query_embeddings):

    vector = vector.reshape(1, -1)

    scores, indices = flat.search(vector, K)

    flat_results[query] = indices[0].tolist()

# --------------------------------------------------
# HNSW benchmark
# --------------------------------------------------

print("\n========================================")
print("HNSW BENCHMARK")
print("========================================")

print(
    f"{'efSearch':>10} "
    f"{'Recall@5':>12} "
    f"{'Avg Search ms':>16}"
)

print("-" * 42)

for ef in ef_values:

    hnsw.hnsw.efSearch = ef

    recalls = []
    latencies = []

    for query, vector in zip(queries, query_embeddings):

        vector = vector.reshape(1, -1)

        start = time.perf_counter()

        scores, indices = hnsw.search(vector, K)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

        exact_ids = set(flat_results[query])
        hnsw_ids = set(indices[0])

        overlap = len(exact_ids.intersection(hnsw_ids))

        recall = overlap / K

        recalls.append(recall)

    avg_recall = np.mean(recalls) * 100
    avg_latency = np.mean(latencies)

    print(
        f"{ef:>10} "
        f"{avg_recall:>11.2f}% "
        f"{avg_latency:>15.3f}"
    )

print("\n========================================")
print("BENCHMARK COMPLETE")
print("========================================")
