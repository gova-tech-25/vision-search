import os
from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Configurable Dataset Directory
ENV_DATASET_DIR = os.getenv("DATASET_DIR", "Flick 30k Dataset for Image Captioning")
POSSIBLE_DATASET_PATHS = [
    Path(ENV_DATASET_DIR),
    BASE_DIR / ENV_DATASET_DIR,
    BASE_DIR / "data" / "flickr30k",
    BASE_DIR / "data" / "Flick 30k Dataset for Image Captioning",
]

DATASET_DIR = None
for path in POSSIBLE_DATASET_PATHS:
    if path.exists() and path.is_dir():
        DATASET_DIR = path.resolve()
        break

if DATASET_DIR is None:
    # Default to expected relative path even if missing for error reporting
    DATASET_DIR = (BASE_DIR / ENV_DATASET_DIR).resolve()

# Artifacts Directory
ARTIFACTS_DIR = (BASE_DIR / "artifacts").resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact file paths
EMBEDDINGS_PATH = ARTIFACTS_DIR / "image_embeddings.npy"
METADATA_PATH = ARTIFACTS_DIR / "image_metadata.json"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
HNSW_INDEX_PATH = ARTIFACTS_DIR / "hnsw.index"
SPLIT_INFO_PATH = ARTIFACTS_DIR / "split_info.json"

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")

# Processing Configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() in ("true", "1", "yes")

if FORCE_CPU or not torch.cuda.is_available():
    DEVICE = "cpu"
else:
    DEVICE = "cuda"

# Retrieval Configuration
RETRIEVAL_BACKEND = os.getenv("RETRIEVAL_BACKEND", "bruteforce")  # 'bruteforce', 'flat', or 'hnsw'
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))

# HNSW Configuration
HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))

# Split sizes for Evaluation
TEST_SPLIT_SIZE = 1000
VAL_SPLIT_SIZE = 1000
