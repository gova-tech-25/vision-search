import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import csv
import time
import numpy as np
import faiss

from src.config import (
    EMBEDDINGS_PATH,
    METADATA_PATH,
    FAISS_INDEX_PATH,
    HNSW_INDEX_PATH,
)

from src.embeddings import EmbeddingGenerator


# ============================================================
# WEEK 3 - HNSW PARAMETER ANALYSIS
# ============================================================

print("========================================")
print("WEEK 3 - HNSW PARAMETER ANALYSIS")
print("========================================")


# ------------------------------------------------------------
# Load existing artifacts
# ------------------------------------------------------------

embeddings = np.load(EMBEDDINGS_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = __import__("json").load(f)

flat = faiss.read_index(str(FAISS_INDEX_PATH))
hnsw = faiss.read_index(str(HNSW_INDEX_PATH))


print(f"\nImages: {len(metadata)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"FlatIP vectors: {flat.ntotal}")
print(f"HNSW vectors: {hnsw.ntotal}")


# ------------------------------------------------------------
# Queries
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

K = 10

# More efSearch values for a denser experimental curve
ef_values = [
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
]


# ------------------------------------------------------------
# Create CLIP text embeddings
# ------------------------------------------------------------

print("\n========================================")
print("GENERATING QUERY EMBEDDINGS")
print("========================================")

generator = EmbeddingGenerator()

query_embeddings = []

for query in queries:

    embedding = generator.generate_text_embeddings(query)

    query_embeddings.append(
        embedding[0]
    )

query_embeddings = np.asarray(
    query_embeddings,
    dtype=np.float32
)

print(
    f"Query embeddings shape: "
    f"{query_embeddings.shape}"
)


# ------------------------------------------------------------
# Compute FlatIP ground truth
# ------------------------------------------------------------

print("\n========================================")
print("COMPUTING FLATIP GROUND TRUTH")
print("========================================")

flat_results = {}

for query, vector in zip(
    queries,
    query_embeddings
):

    vector = vector.reshape(1, -1)

    scores, indices = flat.search(
        vector,
        K
    )

    flat_results[query] = indices[0].tolist()


print("FlatIP ground truth computed.")


# ------------------------------------------------------------
# HNSW parameter experiment
# ------------------------------------------------------------

print("\n========================================")
print("HNSW PARAMETER EXPERIMENT")
print("========================================")

print(
    f"{'efSearch':>10} "
    f"{'Recall@10':>12} "
    f"{'Avg Time ms':>15} "
    f"{'Min Time ms':>15} "
    f"{'Max Time ms':>15}"
)

print("-" * 72)


results = []


for ef in ef_values:

    print(f"\nTesting efSearch = {ef}")

    # Set HNSW search parameter
    hnsw.hnsw.efSearch = ef

    recalls = []
    latencies = []

    for query, vector in zip(
        queries,
        query_embeddings
    ):

        vector = vector.reshape(1, -1)

        # ----------------------------------------------------
        # Measure HNSW search time
        # ----------------------------------------------------

        start = time.perf_counter()

        scores, indices = hnsw.search(
            vector,
            K
        )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latencies.append(
            latency_ms
        )

        # ----------------------------------------------------
        # Calculate Recall@10
        # ----------------------------------------------------

        exact_ids = set(
            flat_results[query]
        )

        hnsw_ids = set(
            indices[0]
        )

        overlap = len(
            exact_ids.intersection(
                hnsw_ids
            )
        )

        recall = overlap / K

        recalls.append(
            recall
        )

    # --------------------------------------------------------
    # Aggregate results
    # --------------------------------------------------------

    avg_recall = (
        np.mean(recalls) * 100
    )

    avg_latency = np.mean(
        latencies
    )

    min_latency = np.min(
        latencies
    )

    max_latency = np.max(
        latencies
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"{ef:>10} "
        f"{avg_recall:>11.2f}% "
        f"{avg_latency:>14.3f} "
        f"{min_latency:>14.3f} "
        f"{max_latency:>14.3f}"
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "efSearch": ef,
        "recall_at_10": round(
            avg_recall,
            4
        ),
        "avg_search_time_ms": round(
            avg_latency,
            4
        ),
        "min_search_time_ms": round(
            min_latency,
            4
        ),
        "max_search_time_ms": round(
            max_latency,
            4
        ),
    })


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

output_file = "week3_results.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "efSearch",
        "recall_at_10",
        "avg_search_time_ms",
        "min_search_time_ms",
        "max_search_time_ms",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ------------------------------------------------------------
# Complete
# ------------------------------------------------------------

print("\n========================================")
print("WEEK 3 ANALYSIS COMPLETE")
print("========================================")

print(
    f"Tested {len(ef_values)} efSearch values."
)

print(
    f"Results saved to: {output_file}"
)