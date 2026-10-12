

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS_DIR
from src.dataset import Flickr30kDataset
from src.retrieval import RetrievalEngine


def collect_queries(split="test", max_queries=100):
    dataset = Flickr30kDataset()
    queries = []

    for image_id in dataset.get_image_ids(split):
        if image_id not in dataset.image_data:
            continue

        captions = dataset.image_data[image_id]["captions"]

        for caption in captions:
            if caption and caption.strip():
                queries.append(caption.strip())

                if max_queries > 0 and len(queries) >= max_queries:
                    return queries

    return queries


def benchmark_hnsw(
    ef_values,
    max_queries=100,
    repeats=5,
    k=10,
    split="test",
    output_path=None,
):
    if repeats < 1 or k < 1:
        raise ValueError("repeats and k must be positive.")

    if not ef_values or any(value < 1 for value in ef_values):
        raise ValueError("All efSearch values must be positive.")

    if len(set(ef_values)) != len(ef_values):
        raise ValueError("efSearch values must be unique.")

    engine = RetrievalEngine()

    if engine.hnsw_index is None:
        raise RuntimeError("HNSW index is unavailable.")

    if engine.faiss_flat_index is None:
        raise RuntimeError("FlatIP index is unavailable.")

    k = min(k, engine.num_images)
    queries = collect_queries(split, max_queries)

    if not queries:
        raise ValueError(f"No caption queries found for split '{split}'.")

    print(f"Dataset index size: {engine.num_images}")
    print(f"Queries: {len(queries)}")
    print(f"Top-k: {k} | Repeats: {repeats}")

    print("\nEncoding text queries once...")
    query_embeddings = []

    for query in queries:
        embedding = engine.generator.generate_text_embeddings(query)
        query_embeddings.append(
            np.ascontiguousarray(embedding, dtype=np.float32)
        )

    print("Computing exact FlatIP ground truth...")
    ground_truth = []

    for embedding in query_embeddings:
        _, indices = engine.faiss_flat_index.search(embedding, k)
        ground_truth.append(
            set(int(i) for i in indices[0] if i >= 0)
        )

    rows = []

    for ef in ef_values:
        engine.hnsw_index.hnsw.efSearch = ef

        latencies = []
        recalls = []

        print(f"\nBenchmarking efSearch={ef}...")

        for embedding, exact_ids in zip(
            query_embeddings, ground_truth
        ):
            # Warm up before measuring this query.
            engine.hnsw_index.search(embedding, k)

            last_indices = None

            for _ in range(repeats):
                start = time.perf_counter_ns()

                _, indices = engine.hnsw_index.search(
                    embedding, k
                )

                elapsed_ms = (
                    time.perf_counter_ns() - start
                ) / 1_000_000

                latencies.append(elapsed_ms)
                last_indices = indices[0]

            approximate_ids = set(
                int(i) for i in last_indices if i >= 0
            )

            if exact_ids:
                recall = (
                    len(approximate_ids & exact_ids)
                    / len(exact_ids)
                )
                recalls.append(recall)

        row = {
            "efSearch": ef,
            "recall_at_10": round(
                float(np.mean(recalls)) * 100, 2
            ),
            "avg_search_time_ms": round(
                float(np.mean(latencies)), 4
            ),
            "std_search_time_ms": round(
                float(np.std(latencies)), 4
            ),
            "p95_search_time_ms": round(
                float(np.percentile(latencies, 95)), 4
            ),
            "min_search_time_ms": round(
                float(np.min(latencies)), 4
            ),
            "max_search_time_ms": round(
                float(np.max(latencies)), 4
            ),
            "num_queries": len(recalls),
            "repeats": repeats,
        }

        rows.append(row)

        print(
            f"Recall@{k}: {row['recall_at_10']}% | "
            f"Mean: {row['avg_search_time_ms']} ms | "
            f"Std: {row['std_search_time_ms']} ms | "
            f"P95: {row['p95_search_time_ms']} ms"
        )

    if output_path is None:
        output_path = (
            ARTIFACTS_DIR / "hnsw_benchmark_results.csv"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output_path}")
    return rows


DEFAULT_BENCHMARK_QUERIES = [
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


def evaluate_queries_comparison(
    queries=None,
    k=10,
    backends=None,
    ef_search=None,
    output_path=None,
):
    """
    Evaluates and compares retrieval on a curated list of natural-language queries
    across multiple backends (FlatIP vs HNSW), saving per-result rankings and latencies.
    """
    if queries is None:
        queries = DEFAULT_BENCHMARK_QUERIES
    if backends is None:
        backends = ["flat", "hnsw"]

    engine = RetrievalEngine()
    rows = []

    print(f"=== Evaluating Query Retrieval Comparison ({len(queries)} queries) ===")
    for query in queries:
        for backend in backends:
            res = engine.search(query=query, k=k, backend=backend, ef_search=ef_search)
            latency = res["latency"]

            for item in res["results"]:
                first_caption = item["captions"][0] if item.get("captions") else ""
                rows.append({
                    "query": query,
                    "backend": backend,
                    "rank": item["rank"],
                    "image_id": item["image_id"],
                    "score": round(item["score"], 4),
                    "image_path": item["image_path"],
                    "caption": first_caption,
                    "search_time_ms": latency["search_ms"],
                    "total_time_ms": latency["total_ms"],
                })

    if output_path is None:
        output_path = ARTIFACTS_DIR / "week3_query_comparison_results.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Query comparison results saved to: {output_path}")

    return rows


def load_benchmark_results(file_path=None):
    """Loads benchmark CSV results from artifacts or week3 results."""
    candidate_paths = [
        Path(file_path) if file_path else None,
        ARTIFACTS_DIR / "hnsw_benchmark_results.csv",
        Path(__file__).resolve().parent.parent / "week3" / "results" / "week3_results.csv",
    ]

    for p in candidate_paths:
        if p and p.exists():
            try:
                import pandas as pd
                return pd.read_csv(p)
            except Exception:
                pass
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark HNSW recall and search latency."
    )

    parser.add_argument(
        "--mode",
        choices=["sweep", "queries"],
        default="sweep",
        help="Run parameter sweep or test query comparison.",
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "val", "test", "full"]
    )
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--ef-values",
        type=int,
        nargs="+",
        default=[16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256],
    )
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.mode == "queries":
        evaluate_queries_comparison(k=args.k, output_path=args.output)
    else:
        benchmark_hnsw(
            ef_values=args.ef_values,
            max_queries=args.max_queries,
            repeats=args.repeats,
            k=args.k,
            split=args.split,
            output_path=args.output,
        )

