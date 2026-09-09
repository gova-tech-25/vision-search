# Text-to-Image Retrieval System using CLIP and Flickr30k

An end-to-end, production-grade multimodal information retrieval system that enables natural language text queries to retrieve semantically relevant images from the **Flickr30k** dataset using **OpenAI's CLIP** (Contrastive Language-Image Pre-training) model and vector similarity search (**Brute-Force Cosine Similarity** and **FAISS Indexing**).

---

## 1. Architecture & Overview

```text
                     FLICKR30K DATASET (31,783 Images)
                                    │
                       CLIP Vision Encoder (ViT-B/32)
                                    │
                    Image Embeddings (31,783 x 512)
                                    │
                          L2 Normalization
                                    │
                   ┌────────────────┴────────────────┐
                   ▼                                 ▼
         artifacts/image_embeddings.npy   artifacts/faiss.index
                   │                                 │
                   └────────────────┬────────────────┘
                                    │
  USER TEXT QUERY ──► CLIP Text Encoder ──► L2 Normalize ──► Similarity Engine
  ("A dog running in grass")                                     │
                                                         Top-K Ranked Results
                                                                 │
                                                       Streamlit Web UI & Metrics
```

### Key Technical Concepts
1. **Shared Embedding Space**: Pretrained CLIP maps both text descriptions and visual features into a unified 512-dimensional vector space $\mathbb{R}^{512}$.
2. **Offline Indexing**: All 31,783 Flickr30k images are encoded and L2-normalized once offline. Embeddings are stored on disk (`image_embeddings.npy` and `faiss.index`) so online query processing avoids recomputing image features.
3. **Vector Similarity**: Query text $q$ and images $x_i$ are compared using Cosine Similarity. Since all embeddings are unit-normalized ($\|v\|_2 = 1.0$), Cosine Similarity simplifies to an efficient dot product:
   $$\text{Sim}(q, x_i) = \frac{q \cdot x_i}{\|q\|_2 \|x_i\|_2} = q \cdot x_i$$
4. **Held-Out Evaluation**: Strict dataset splitting ensures zero image leakage between evaluation queries and held-out test targets.

---

## 2. Dataset — Flickr30k

The system operates on the **Flickr30k** dataset containing 31,783 high-resolution images, each paired with 5 human-annotated natural language captions (~158,917 captions total).

### Directory Structure
```text
Flick 30k Dataset for Image Captioning/
├── Images/
│   ├── 1000092795.jpg
│   ├── 10002456.jpg
│   └── ...
└── captions.txt
```

### Dataset Splits
To guarantee rigorous quantitative evaluation without data leakage:
- **Test Set**: 1,000 held-out images (~5,000 caption queries).
- **Validation Set**: 1,000 held-out images.
- **Train / Indexing Set**: Remaining 29,783 images (or full collection for interactive search).

---

## 3. Pretrained Model — OpenAI CLIP

The baseline system utilizes Hugging Face's implementation of `openai/clip-vit-base-patch32`:
- **Vision Encoder**: Vision Transformer (ViT-B/32) mapping $224 \times 224$ images to 512-dimensional vectors.
- **Text Encoder**: Transformer text encoder converting tokenized captions (up to 77 tokens) to 512-dimensional vectors.
- **No Fine-Tuning**: Pretrained zero-shot CLIP capabilities are leveraged directly.

---

## 4. Quantitative Evaluation Metrics

The system includes a dedicated evaluation framework (`src/evaluation.py`) computing the following metrics:

| Metric | Definition / Formula | Explanation |
| :--- | :--- | :--- |
| **Recall@1 (R@1)** | $\frac{\sum \mathbb{I}(\text{Rank} = 1)}{N}$ | Percentage of queries where ground-truth image is retrieved at Rank 1. |
| **Recall@5 (R@5)** | $\frac{\sum \mathbb{I}(\text{Rank} \le 5)}{N}$ | Percentage of queries where ground-truth image is in Top-5. |
| **Recall@10 (R@10)** | $\frac{\sum \mathbb{I}(\text{Rank} \le 10)}{N}$ | Percentage of queries where ground-truth image is in Top-10. |
| **Precision@5 (P@5)** | $\frac{\sum \text{Rel}_5}{5 \cdot N}$ | Precision of top-5 retrieved items. |
| **Precision@10 (P@10)** | $\frac{\sum \text{Rel}_{10}}{10 \cdot N}$ | Precision of top-10 retrieved items. |
| **MRR** | $\frac{1}{N} \sum \frac{1}{\text{Rank}}$ | Mean Reciprocal Rank across all evaluation queries. |
| **Latency** | $\text{ms / query}$ | Average text encoding time, search time, and total end-to-end query latency. |

---

## 5. Quick Start Guide

### Step 1: Clone Repository & Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

### Step 2: Dataset Placement
Ensure Flickr30k dataset is placed at `Flick 30k Dataset for Image Captioning/` or configure via `.env`:
```env
DATASET_DIR="Flick 30k Dataset for Image Captioning"
```

### Step 3: Run Unit Tests
Verify model initialization, embedding normalization, retrieval ordering, and metrics calculation:
```bash
python -m pytest tests/
```

### Step 4: Build Image Embeddings Index
Build offline persistent embeddings for Flickr30k images:
```bash
python -m src.index --split full --batch-size 64
```

### Step 5: Run Quantitative Evaluation
Evaluate retrieval performance on held-out test split:
```bash
python -m src.evaluation --split test --compare
```

### Step 6: Launch Interactive Streamlit UI
Start web app for interactive text query search:
```bash
streamlit run app.py
```

---

## 6. Project Structure

```text
clip-image-retrieval/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── app.py                   # Streamlit Web Application
│
├── data/                    # Dataset directory
├── artifacts/               # Generated embeddings & vector indices
│   ├── image_embeddings.npy
│   ├── image_metadata.json
│   ├── faiss.index
│   └── split_info.json
│
├── src/                     # Core system modules
│   ├── __init__.py
│   ├── config.py            # Global settings & env configs
│   ├── dataset.py           # Flickr30k parser & split manager
│   ├── model.py             # CLIP model singleton loader
│   ├── embeddings.py        # Feature extraction & L2 normalization
│   ├── index.py             # Offline indexing & FAISS builder
│   ├── retrieval.py         # Vector similarity search engine
│   ├── evaluation.py        # Benchmark metrics & latency recorder
│   └── utils.py             # Timing context managers & normalizers
│
└── tests/                   # Unit test suite
    ├── __init__.py
    ├── test_embeddings.py
    ├── test_retrieval.py
    └── test_evaluation.py
```

---

## 7. Example Queries

Try searching with these queries in the Streamlit UI:
1. `"A dog running through a grassy field"`
2. `"A child playing in a wooden playhouse"`
3. `"A group of people sitting at a cafe table"`
4. `"A person skiing down a snowy mountain slope"`
5. `"Two men working on a construction site wearing hard hats"`

---

## 8. System Benchmarks (Brute Force vs. FAISS)

*Empirical benchmarking results evaluated on 200 held-out Flickr30k test captions:*

| Metric | Brute Force Baseline | FAISS Vector Search |
| :--- | :--- | :--- |
| **Recall@1 (R@1)** | **61.00%** | **61.00%** |
| **Recall@5 (R@5)** | **82.50%** | **82.50%** |
| **Recall@10 (R@10)** | **90.00%** | **90.00%** |
| **Precision@5 (P@5)** | **16.50%** | **16.50%** |
| **Precision@10 (P@10)** | **9.00%** | **9.00%** |
| **Mean Reciprocal Rank (MRR)** | **0.6984** | **0.6984** |
| **Avg Text Encoding Latency** | `102.24 ms` | `38.35 ms` |
| **Avg Vector Search Latency** | `0.66 ms` | `0.35 ms` |
| **Avg Total Latency** | `102.89 ms` | `38.70 ms` |

> [!NOTE]
> FAISS achieves **nearly 2x faster vector search latency (0.35 ms vs 0.66 ms)** with identical top-K retrieval accuracy because `IndexFlatIP` computes exact inner products over normalized 512-d embeddings using C++ SIMD optimizations.

---

## 9. Limitations & Future Work

### Limitations
- **Fine-Grained Details**: Pretrained CLIP ViT-B/32 sometimes struggles with complex spatial relationships or exact object counts (e.g. "exactly 3 red cars").
- **Dataset Bias**: Retrieval quality is bounded by Flickr30k domain distribution (everyday outdoor/social scenes).

### Future Work
- **Advanced FAISS Indexing**: Implement `IndexIVFFlat` or `IndexHNSW` for sub-linear time approximate nearest neighbor search at scale ($>1\text{M}$ images).
- **Fine-Tuning CLIP**: Fine-tune vision and text projection layers on Flickr30k using InfoNCE contrastive loss.
- **Larger Backbones**: Support `openai/clip-vit-large-patch14` for higher visual accuracy.
- **Hybrid Retrieval**: Combine dense CLIP vector embeddings with sparse keyword search (BM25) for hybrid search.
