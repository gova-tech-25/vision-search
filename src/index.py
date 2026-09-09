import argparse
import json
import numpy as np
import faiss
from pathlib import Path
from src.config import (
    EMBEDDINGS_PATH,
    METADATA_PATH,
    FAISS_INDEX_PATH,
    HNSW_INDEX_PATH,
    BATCH_SIZE,
    DEVICE,
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
)
from src.dataset import Flickr30kDataset
from src.embeddings import EmbeddingGenerator
from src.utils import verify_l2_norm

def build_index(split: str = "full", batch_size: int = BATCH_SIZE, force: bool = False):
    """
    Offline indexing pipeline for Flickr30k dataset.
    Generates image embeddings, normalizes them, and builds persistent numpy,
    FAISS FlatIP (exact), and FAISS HNSW (approximate) indices.
    """
    print(f"=== Flickr30k Offline Image Indexing Pipeline ===")
    print(f"Target split: '{split}' | Device: '{DEVICE}' | Batch size: {batch_size}")

    if not force and EMBEDDINGS_PATH.exists() and METADATA_PATH.exists():
        print(f"Existing embeddings index found at '{EMBEDDINGS_PATH}'. Use --force to rebuild.")
        return

    # Load dataset
    dataset = Flickr30kDataset()
    metadata = dataset.get_metadata_for_split(split)

    if not metadata:
        raise ValueError(f"No metadata found for split '{split}'. Check dataset loading.")

    image_paths = [m["image_path"] for m in metadata]
    print(f"Found {len(image_paths)} images to index for split '{split}'.")

    # Generate embeddings (returns successful indices to handle failed images)
    generator = EmbeddingGenerator()
    embeddings, successful_indices = generator.generate_image_embeddings(
        image_paths, batch_size=batch_size
    )

    # Filter metadata to only successfully embedded images
    failed_count = len(image_paths) - len(successful_indices)
    if failed_count > 0:
        print(f"Warning: {failed_count} images failed to load/encode and were skipped.")
        metadata = [metadata[i] for i in successful_indices]
        # Re-index the metadata entries sequentially
        for new_idx, m in enumerate(metadata):
            m["index"] = new_idx

    # Strict alignment validation
    assert embeddings.shape[0] == len(metadata), (
        f"FATAL: Embedding/metadata alignment failure! "
        f"Embeddings: {embeddings.shape[0]}, Metadata: {len(metadata)}. "
        f"This should never happen - please report this bug."
    )
    print(f"Alignment verified: {embeddings.shape[0]} embeddings <-> {len(metadata)} metadata records.")

    # Verify L2 normalization
    is_normalized = verify_l2_norm(embeddings)
    print(f"L2 Normalization check: {'PASSED' if is_normalized else 'FAILED'}")

    # Save NumPy embeddings matrix
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved image embeddings matrix ({embeddings.shape}): '{EMBEDDINGS_PATH.name}'")

    # Save metadata mapping
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved image metadata ({len(metadata)} records): '{METADATA_PATH.name}'")

    # --- Build FAISS FlatIP Index (Exact Search) ---
    dimension = embeddings.shape[1]
    faiss_flat = faiss.IndexFlatIP(dimension)
    faiss_flat.add(embeddings)
    faiss.write_index(faiss_flat, str(FAISS_INDEX_PATH))
    print(f"Saved FAISS FlatIP index ({faiss_flat.ntotal} vectors, d={dimension}): '{FAISS_INDEX_PATH.name}'")

    # --- Build FAISS HNSW Index (Approximate Search) ---
    print(f"Building HNSW index (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION})...")
    hnsw_index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    hnsw_index.add(embeddings)
    faiss.write_index(hnsw_index, str(HNSW_INDEX_PATH))
    print(f"Saved FAISS HNSW index ({hnsw_index.ntotal} vectors, d={dimension}): '{HNSW_INDEX_PATH.name}'")

    # Final triple-alignment verification
    assert faiss_flat.ntotal == len(metadata) == embeddings.shape[0], (
        f"FATAL: Final artifact count mismatch! "
        f"FlatIP: {faiss_flat.ntotal}, HNSW: {hnsw_index.ntotal}, "
        f"Metadata: {len(metadata)}, Embeddings: {embeddings.shape[0]}"
    )

    print(f"\n=== Indexing pipeline completed successfully! ===")
    print(f"    Embeddings : {embeddings.shape}")
    print(f"    Metadata   : {len(metadata)} records")
    print(f"    FlatIP     : {faiss_flat.ntotal} vectors")
    print(f"    HNSW       : {hnsw_index.ntotal} vectors (M={HNSW_M})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Flickr30k Image Index using CLIP")
    parser.add_argument("--split", type=str, default="full", help="Dataset split to index (full, train, val, test)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for image processing")
    parser.add_argument("--force", action="store_true", help="Force rebuild existing index")
    args = parser.parse_args()

    build_index(split=args.split, batch_size=args.batch_size, force=args.force)
