import sys
from pathlib import Path
import json
import csv
import time
import numpy as np
import faiss

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval import RetrievalEngine
# ============================================================
# WEEK 3 - RETRIEVAL EVALUATION
# Full Flickr30k Dataset
# ============================================================

print("========================================")
print("WEEK 3 - RETRIEVAL EVALUATION")
print("========================================")

# ------------------------------------------------------------
# Initialize retrieval engine
# ------------------------------------------------------------

engine = RetrievalEngine()

print(f"\nTotal images in dataset: {engine.num_images}")

# ------------------------------------------------------------
# Natural-language test queries
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

backends = [
    "flat",
    "hnsw",
]

# ------------------------------------------------------------
# CSV output
# ------------------------------------------------------------

output_file = "week3_retrieval_results.csv"

rows = []

# ------------------------------------------------------------
# Run evaluation
# ------------------------------------------------------------

for query in queries:

    print("\n========================================")
    print("QUERY:", query)
    print("========================================")

    for backend in backends:

        print(f"\nBackend: {backend}")

        result = engine.search(
            query=query,
            k=K,
            backend=backend
        )

        latency = result["latency"]

        print("Latency:")
        print(f"  Text encoding: {latency['text_encoding_ms']:.3f} ms")
        print(f"  Search:        {latency['search_ms']:.3f} ms")
        print(f"  Total:         {latency['total_ms']:.3f} ms")

        print("\nTop results:")

        for item in result["results"]:

            caption = (
                item["captions"][0]
                if item["captions"]
                else "No caption"
            )

            print(
                f"{item['rank']:2d}. "
                f"{item['image_id']} | "
                f"score={item['score']:.4f} | "
                f"{caption}"
            )

            rows.append({
                "query": query,
                "backend": backend,
                "rank": item["rank"],
                "image_id": item["image_id"],
                "score": item["score"],
                "image_path": item["image_path"],
                "caption": caption,
                "search_time_ms": latency["search_ms"],
                "total_time_ms": latency["total_ms"],
            })


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

fieldnames = [
    "query",
    "backend",
    "rank",
    "image_id",
    "score",
    "image_path",
    "caption",
    "search_time_ms",
    "total_time_ms",
]

with open(output_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(f"Results saved to: {output_file}")