import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
from src.dataset import Flickr30kDataset
from src.retrieval import RetrievalEngine
from src.utils import Timer

def evaluate_retrieval(
    split: str = "test",
    max_queries: int = 1000,
    backend: str = "bruteforce",
    engine: RetrievalEngine = None,
) -> Dict[str, Any]:
    """
    Evaluates Text-to-Image retrieval performance on held-out dataset split.

    Metrics:
        - Recall@1, Recall@5, Recall@10
        - Precision@5, Precision@10
        - Mean Reciprocal Rank (MRR)
        - Latency (Encoding time, Search time, Total time)
    """
    if engine is None:
        engine = RetrievalEngine()

    dataset = Flickr30kDataset()
    # Keep list order (not set) so max_queries sampling is deterministic
    split_image_ids = dataset.get_image_ids(split)

    # Collect test queries: tuple (caption, target_image_id)
    test_queries = []
    for img_id in split_image_ids:
        if img_id in dataset.image_data:
            for caption in dataset.image_data[img_id]["captions"]:
                test_queries.append((caption, img_id))

    if max_queries and max_queries < len(test_queries):
        # Sample deterministically for speed if max_queries specified
        test_queries = test_queries[:max_queries]

    total_queries = len(test_queries)
    print(f"\n--- Starting Quantitative Evaluation on split '{split}' ---")
    print(f"Backend: '{backend}' | Total Queries: {total_queries} | Index Size: {engine.num_images}")

    r1_count = 0
    r5_count = 0
    r10_count = 0
    p5_sum = 0.0
    p10_sum = 0.0
    reciprocal_ranks = []

    encoding_times = []
    search_times = []
    total_times = []

    k_max = 10

    for query, target_img_id in tqdm(test_queries, desc=f"Evaluating ({backend})"):
        # Measure query retrieval
        res = engine.search(query, k=k_max, backend=backend)

        encoding_times.append(res["latency"]["text_encoding_ms"])
        search_times.append(res["latency"]["search_ms"])
        total_times.append(res["latency"]["total_ms"])

        retrieved_ids = [r["image_id"] for r in res["results"]]

        # Rank of first relevant result (1-indexed)
        if target_img_id in retrieved_ids:
            rank = retrieved_ids.index(target_img_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            rank = None
            reciprocal_ranks.append(0.0)

        # Recall@K
        if rank is not None:
            if rank == 1:
                r1_count += 1
            if rank <= 5:
                r5_count += 1
                p5_sum += 1.0 / 5.0  # 1 relevant image in top 5
            if rank <= 10:
                r10_count += 1
                p10_sum += 1.0 / 10.0  # 1 relevant image in top 10

    metrics = {
        "split": split,
        "backend": backend,
        "total_queries": total_queries,
        "recall@1": round(r1_count / total_queries, 4),
        "recall@5": round(r5_count / total_queries, 4),
        "recall@10": round(r10_count / total_queries, 4),
        "precision@5": round(p5_sum / total_queries, 4),
        "precision@10": round(p10_sum / total_queries, 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "latency": {
            "avg_text_encoding_ms": round(float(np.mean(encoding_times)), 2),
            "avg_search_ms": round(float(np.mean(search_times)), 2),
            "avg_total_ms": round(float(np.mean(total_times)), 2),
        },
    }

    return metrics

def print_evaluation_report(metrics: Dict[str, Any]):
    """Pretty prints evaluation metrics report."""
    print("\n" + "=" * 55)
    print(f"        RETRIEVAL EVALUATION REPORT ({metrics['backend'].upper()})")
    print("=" * 55)
    print(f"Split Evaluated      : {metrics['split']}")
    print(f"Total Test Queries   : {metrics['total_queries']}")
    print("-" * 55)
    print(f"Recall@1  (R@1)      : {metrics['recall@1'] * 100:.2f}%")
    print(f"Recall@5  (R@5)      : {metrics['recall@5'] * 100:.2f}%")
    print(f"Recall@10 (R@10)     : {metrics['recall@10'] * 100:.2f}%")
    print(f"Precision@5  (P@5)   : {metrics['precision@5'] * 100:.2f}%")
    print(f"Precision@10 (P@10)  : {metrics['precision@10'] * 100:.2f}%")
    print(f"Mean Reciprocal Rank : {metrics['mrr']:.4f}")
    print("-" * 55)
    print(f"Avg Text Encoding    : {metrics['latency']['avg_text_encoding_ms']} ms")
    print(f"Avg Vector Search    : {metrics['latency']['avg_search_ms']} ms")
    print(f"Avg Total Latency    : {metrics['latency']['avg_total_ms']} ms")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP Text-to-Image Retrieval on Flickr30k")
    parser.add_argument("--split", type=str, default="test", help="Dataset split for evaluation (test, val)")
    parser.add_argument("--max-queries", type=int, default=1000, help="Maximum number of test queries")
    parser.add_argument("--backend", type=str, default="bruteforce",
                        help="Backend to evaluate: bruteforce, flat, or hnsw")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all available backends (Brute Force, FlatIP, HNSW)")
    args = parser.parse_args()

    engine = RetrievalEngine()

    if args.compare:
        backends = ["bruteforce", "flat"]
        if engine.hnsw_index is not None:
            backends.append("hnsw")
        else:
            print("Note: HNSW index not available, skipping HNSW evaluation.")

        for be in backends:
            m = evaluate_retrieval(
                split=args.split, max_queries=args.max_queries,
                backend=be, engine=engine
            )
            print_evaluation_report(m)
    else:
        metrics = evaluate_retrieval(
            split=args.split, max_queries=args.max_queries,
            backend=args.backend, engine=engine
        )
        print_evaluation_report(metrics)

