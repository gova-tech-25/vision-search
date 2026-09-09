import json
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.config import (
    EMBEDDINGS_PATH,
    METADATA_PATH,
    FAISS_INDEX_PATH,
    HNSW_INDEX_PATH,
    RETRIEVAL_BACKEND,
    DEFAULT_K,
    HNSW_EF_SEARCH,
    DATASET_DIR,
)
from src.embeddings import EmbeddingGenerator
from src.utils import Timer

VALID_BACKENDS = {"bruteforce", "flat", "hnsw"}

class RetrievalEngine:
    """
    Multimodal Text-to-Image Retrieval Engine supporting:
        - Brute-Force cosine similarity (dot product of unit vectors)
        - FAISS FlatIP - exact nearest-neighbor search
        - FAISS HNSW - approximate nearest-neighbor search
    """

    def __init__(self, generator: Optional[EmbeddingGenerator] = None):
        self._check_artifacts_exist()

        print("Loading persistent image index and metadata...")
        self.embeddings = np.load(EMBEDDINGS_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.num_images = len(self.metadata)

        # Validate embedding/metadata alignment
        if self.embeddings.shape[0] != self.num_images:
            raise ValueError(
                f"Embedding/metadata count mismatch: "
                f"{self.embeddings.shape[0]} embeddings vs {self.num_images} metadata records. "
                f"Please rebuild the index with: python -m src.index --force"
            )
        print(f"Loaded index containing {self.num_images} image vectors ({self.embeddings.shape}).")

        # Load FAISS FlatIP index if available
        self.faiss_flat_index = None
        if FAISS_INDEX_PATH.exists():
            try:
                self.faiss_flat_index = faiss.read_index(str(FAISS_INDEX_PATH))
                print(f"Loaded FAISS FlatIP index ({self.faiss_flat_index.ntotal} vectors).")
            except Exception as e:
                print(f"Warning: Failed to load FAISS FlatIP index: {e}")

        # Load FAISS HNSW index if available
        self.hnsw_index = None
        if HNSW_INDEX_PATH.exists():
            try:
                self.hnsw_index = faiss.read_index(str(HNSW_INDEX_PATH))
                self.hnsw_index.hnsw.efSearch = HNSW_EF_SEARCH
                print(f"Loaded FAISS HNSW index ({self.hnsw_index.ntotal} vectors, efSearch={HNSW_EF_SEARCH}).")
            except Exception as e:
                print(f"Warning: Failed to load FAISS HNSW index: {e}")

        self._rebase_image_paths()

        self.generator = generator or EmbeddingGenerator()

    def _rebase_image_paths(self):
        # ponytail: metadata bakes absolute paths from the build machine (e.g. Windows C:\);
        # if dead, re-resolve against the local dataset dir. Proper fix is storing relative
        # paths at index time + full reindex.
        sample = Path(self.metadata[0]["image_path"])
        if sample.exists():
            return
        images_dir = None
        for cand in (DATASET_DIR / "Images", DATASET_DIR / "images", DATASET_DIR / "flickr30k_images", DATASET_DIR):
            if cand.is_dir() and any(cand.glob("*.jpg")):
                images_dir = cand.resolve()
                break
        if images_dir is None:
            print(f"Warning: stored image paths are invalid and no dataset found under '{DATASET_DIR}'.")
            return
        for m in self.metadata:
            m["image_path"] = str(images_dir / m["image_id"])
        print(f"Rebased {len(self.metadata)} image paths to '{images_dir}'.")

    def _check_artifacts_exist(self):
        """Validates index presence before instantiation."""
        if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"\n[ERROR] Image index files not found!\n"
                f"Missing: '{EMBEDDINGS_PATH}' or '{METADATA_PATH}'.\n"
                f"Please run the offline indexing script first:\n"
                f"    python -m src.index\n"
            )

    def search(
        self, query: str, k: int = DEFAULT_K, backend: str = RETRIEVAL_BACKEND,
        ef_search: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes text-to-image similarity search.

        Args:
            query: Natural language text query.
            k: Top-K results to retrieve.
            backend: Search backend ('bruteforce', 'flat', or 'hnsw').
            ef_search: Optional HNSW efSearch override for tuning experiments.

        Returns:
            Dict containing retrieved results, scores, and detailed latency metrics.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        query = query.strip()

        # Validate backend
        backend_lower = backend.lower()
        if backend_lower not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend '{backend}'. Valid options: {sorted(VALID_BACKENDS)}"
            )

        # Validate k
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")
        k = min(k, self.num_images)

        if self.num_images == 0:
            return {
                "query": query, "backend": backend_lower, "k": k,
                "latency": {"text_encoding_ms": 0, "search_ms": 0, "total_ms": 0},
                "results": [],
            }

        # 1. Encode text query
        with Timer() as timer_enc:
            text_embedding = self.generator.generate_text_embeddings(query)
        encoding_time_ms = timer_enc.interval_ms

        # 2. Vector Similarity Search
        with Timer() as timer_search:
            if backend_lower == "hnsw":
                if self.hnsw_index is None:
                    raise RuntimeError(
                        "HNSW index not available. "
                        "Rebuild the index with: python -m src.index --force"
                    )
                # Optionally override efSearch for tuning experiments
                if ef_search is not None:
                    self.hnsw_index.hnsw.efSearch = ef_search
                scores, indices = self.hnsw_index.search(text_embedding, k)
                top_indices = indices[0]
                top_scores = scores[0]
            elif backend_lower == "flat":
                if self.faiss_flat_index is None:
                    raise RuntimeError(
                        "FAISS FlatIP index not available. "
                        "Rebuild the index with: python -m src.index --force"
                    )
                scores, indices = self.faiss_flat_index.search(text_embedding, k)
                top_indices = indices[0]
                top_scores = scores[0]
            else:
                # Brute force cosine similarity via dot product of unit vectors
                # text_embedding: (1, 512), embeddings: (N, 512)
                sims = np.dot(self.embeddings, text_embedding.T).squeeze(-1)
                # Sort descending
                top_indices = np.argsort(-sims)[:k]
                top_scores = sims[top_indices]

        search_time_ms = timer_search.interval_ms
        total_time_ms = encoding_time_ms + search_time_ms

        # 3. Assemble Top-K results, filtering out invalid FAISS indices (-1)
        results = []
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
            idx = int(idx)
            # FAISS returns -1 when the index has fewer vectors than k
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "image_id": meta["image_id"],
                "image_path": meta["image_path"],
                "captions": meta.get("captions", []),
                "index": idx,
            })

        # Re-rank after filtering invalid indices
        for i, r in enumerate(results, start=1):
            r["rank"] = i

        return {
            "query": query,
            "backend": backend_lower,
            "k": k,
            "latency": {
                "text_encoding_ms": round(encoding_time_ms, 3),
                "search_ms": round(search_time_ms, 3),
                "total_ms": round(total_time_ms, 3),
            },
            "results": results,
        }
