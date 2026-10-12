"""
CLIP-Retrieve: Production-Grade Multimodal Text-to-Image Retrieval System
Trained and evaluated on Flickr30k with FAISS FlatIP and HNSW approximate indexing.
"""

__version__ = "1.3.0"

from src.retrieval import RetrievalEngine
from src.dataset import Flickr30kDataset
from src.embeddings import EmbeddingGenerator
from src.evaluation import evaluate_retrieval
from src.benchmark import benchmark_hnsw, evaluate_queries_comparison

__all__ = [
    "RetrievalEngine",
    "Flickr30kDataset",
    "EmbeddingGenerator",
    "evaluate_retrieval",
    "benchmark_hnsw",
    "evaluate_queries_comparison",
]

