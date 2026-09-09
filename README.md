<div align="center">

# 🔍 CLIP-Retrieve

### Production-Grade Text-to-Image Retrieval on Flickr30k

*Natural language queries → semantically ranked images, powered by OpenAI CLIP and vector similarity search*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CLIP-red.svg)](https://github.com/openai/CLIP)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

[Overview](#1-architecture--overview) •
[Dataset](#2-dataset--flickr30k) •
[Model](#3-pretrained-model--openai-clip) •
[Metrics](#4-quantitative-evaluation-metrics) •
[Quick Start](#5-quick-start-guide) •
[Benchmarks](#8-system-benchmarks-brute-force-vs-faiss) •
[Roadmap](#9-limitations--future-work)

</div>

---

## Overview

**CLIP-Retrieve** is an end-to-end multimodal information retrieval system that maps free-form text queries directly onto the **Flickr30k** image corpus, without any task-specific fine-tuning. It uses OpenAI's **CLIP** (Contrastive Language-Image Pre-training) to embed text and images into a shared vector space, then serves top-K results via a brute-force cosine similarity engine or a **FAISS**-accelerated index — with a full quantitative evaluation harness and an interactive Streamlit front end.

**Highlights**
- 🧠 Zero-shot semantic search — no fine-tuning required
- ⚡ Sub-40ms end-to-end query latency with FAISS `IndexFlatIP`
- 📊 Rigorous held-out evaluation (Recall@K, Precision@K, MRR, latency)
- 🧪 Unit-tested core modules (embeddings, retrieval, evaluation)
- 🖥️ Interactive Streamlit UI for live text-to-image search

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
                   ▼                                  ▼
         artifacts/image_embeddings.npy   artifacts/faiss.index
                   │                                  │
                   └────────────────┬─────────────────┘
                                    │
  USER TEXT QUERY ──► CLIP Text Encoder ──► L2 Normalize ──► Similarity Engine
  ("A dog running in grass")                                     │
                                                          Top-K Ranked Results
                                                                  │
                                                        Streamlit Web UI & Metrics
```

### Key Technical Concepts

| Concept | Description |
| :--- | :--- |
| **Shared Embedding Space** | Pretrained CLIP maps both text and images into a unified 512-dimensional vector space $\mathbb{R}^{512}$. |
| **Offline Indexing** | All 31,783 images are encoded and L2-normalized once, offline. Embeddings persist to disk (`image_embeddings.npy`, `faiss.index`) so online queries never recompute image features. |
| **Vector Similarity** | Query $q$ and image $x_i$ are compared via cosine similarity. Since all vectors are unit-normalized ($\lVert v \rVert_2 = 1$), this reduces to a dot product: $\text{Sim}(q, x_i) = \dfrac{q \cdot x_i}{\lVert q \rVert_2 \lVert x_i \rVert_2} = q \cdot x_i$ |
| **Held-Out Evaluation** | Strict train/val/test splitting guarantees zero image leakage between evaluation queries and retrieval targets. |

---

## 2. Dataset — Flickr30k

The system operates on **Flickr30k**: 31,783 high-resolution images, each paired with 5 human-annotated captions (~158,917 captions total).

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

Splitting is enforced at the **image level** (not caption level) to prevent leakage:

| Split | Images | Captions | Purpose |
| :--- | :--- | :--- | :--- |
| **Test** | 1,000 | ~5,000 | Held-out final evaluation |
| **Validation** | 1,000 | ~5,000 | Hyperparameter / config tuning |
| **Train / Index** | 29,783 | ~148,917 | Indexing corpus (or full set for interactive search) |

---

## 3. Pretrained Model — OpenAI CLIP

Backed by Hugging Face's implementation of `openai/clip-vit-base-patch32`:

| Component | Details |
| :--- | :--- |
| **Vision Encoder** | ViT-B/32 Vision Transformer — maps $224 \times 224$ images → 512-d vectors |
| **Text Encoder** | Transformer encoder — maps tokenized captions (≤77 tokens) → 512-d vectors |
| **Fine-Tuning** | None — leverages CLIP's pretrained zero-shot representations directly |

---

## 4. Quantitative Evaluation Metrics

`src/evaluation.py` computes a full retrieval benchmark suite:

| Metric | Formula | Description |
| :--- | :--- | :--- |
| **Recall@1** | $\dfrac{\sum \mathbb{I}(\text{Rank}=1)}{N}$ | % of queries where the ground-truth image ranks 1st |
| **Recall@5** | $\dfrac{\sum \mathbb{I}(\text{Rank}\le 5)}{N}$ | % of queries where ground-truth is in the Top-5 |
| **Recall@10** | $\dfrac{\sum \mathbb{I}(\text{Rank}\le 10)}{N}$ | % of queries where ground-truth is in the Top-10 |
| **Precision@5** | $\dfrac{\sum \text{Rel}_5}{5N}$ | Precision of the top-5 retrieved items |
| **Precision@10** | $\dfrac{\sum \text{Rel}_{10}}{10N}$ | Precision of the top-10 retrieved items |
| **MRR** | $\dfrac{1}{N}\sum \dfrac{1}{\text{Rank}}$ | Mean Reciprocal Rank across all queries |
| **Latency** | ms/query | Text encoding, vector search, and end-to-end timing |

---

## 5. Quick Start Guide

### Prerequisites
- Python 3.9+
- ~4 GB disk space for the Flickr30k dataset and generated embeddings
- (Optional) CUDA-capable GPU for faster indexing

### Step 1 — Clone & Install

```bash
git clone https://github.com/<your-username>/clip-image-retrieval.git
cd clip-image-retrieval
pip install -r requirements.txt
```

### Step 2 — Dataset Placement

Place the Flickr30k dataset at `Flick 30k Dataset for Image Captioning/`, or point to a custom path via `.env`:

```env
DATASET_DIR="Flick 30k Dataset for Image Captioning"
```

### Step 3 — Run Unit Tests

Validate model initialization, embedding normalization, retrieval ordering, and metrics logic:

```bash
python -m pytest tests/ -v
```

### Step 4 — Build the Image Index

Encode and persist embeddings for all Flickr30k images:

```bash
python -m src.index --split full --batch-size 64
```

### Step 5 — Run Quantitative Evaluation

Benchmark retrieval performance on the held-out test split:

```bash
python -m src.evaluation --split test --compare
```

### Step 6 — Launch the Interactive UI

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
├── app.py                    # Streamlit web application
│
├── data/                     # Dataset directory
├── artifacts/                # Generated embeddings & vector indices
│   ├── image_embeddings.npy
│   ├── image_metadata.json
│   ├── faiss.index
│   └── split_info.json
│
├── src/                       # Core system modules
│   ├── __init__.py
│   ├── config.py              # Global settings & env configs
│   ├── dataset.py             # Flickr30k parser & split manager
│   ├── model.py                # CLIP model singleton loader
│   ├── embeddings.py          # Feature extraction & L2 normalization
│   ├── index.py                # Offline indexing & FAISS builder
│   ├── retrieval.py            # Vector similarity search engine
│   ├── evaluation.py          # Benchmark metrics & latency recorder
│   └── utils.py                # Timing context managers & normalizers
│
└── tests/                      # Unit test suite
    ├── __init__.py
    ├── test_embeddings.py
    ├── test_retrieval.py
    └── test_evaluation.py
```

---

## 7. Example Queries

Try these in the Streamlit UI:

1. *"A dog running through a grassy field"*
2. *"A child playing in a wooden playhouse"*
3. *"A group of people sitting at a cafe table"*
4. *"A person skiing down a snowy mountain slope"*
5. *"Two men working on a construction site wearing hard hats"*

---

## 8. System Benchmarks (Brute Force vs. FAISS)

*Evaluated on 200 held-out Flickr30k test captions.*

| Metric | Brute Force Baseline | FAISS Vector Search |
| :--- | :---: | :---: |
| Recall@1 | 61.00% | 61.00% |
| Recall@5 | 82.50% | 82.50% |
| Recall@10 | 90.00% | 90.00% |
| Precision@5 | 16.50% | 16.50% |
| Precision@10 | 9.00% | 9.00% |
| Mean Reciprocal Rank (MRR) | 0.6984 | 0.6984 |
| Avg. Text Encoding Latency | 102.24 ms | 38.35 ms |
| Avg. Vector Search Latency | 0.66 ms | 0.35 ms |
| **Avg. Total Latency** | **102.89 ms** | **38.70 ms** |

> **Note:** FAISS achieves **~2x faster vector search** (0.35 ms vs. 0.66 ms) with **identical retrieval accuracy**, since `IndexFlatIP` computes exact inner products over normalized 512-d embeddings using SIMD-optimized C++ routines — no accuracy is traded for the speedup.

---

## 9. Limitations & Future Work

### Limitations
- **Fine-Grained Details** — Pretrained CLIP ViT-B/32 struggles with complex spatial relationships or exact object counts (e.g., "exactly 3 red cars").
- **Dataset Bias** — Retrieval quality is bounded by Flickr30k's domain distribution (everyday outdoor/social scenes).

### Roadmap
- [ ] **Approximate NN Indexing** — `IndexIVFFlat` / `IndexHNSW` for sub-linear search at scale (>1M images)
- [ ] **CLIP Fine-Tuning** — Fine-tune vision/text projection layers on Flickr30k with InfoNCE contrastive loss
- [ ] **Larger Backbones** — Support `openai/clip-vit-large-patch14` for higher visual fidelity
- [ ] **Hybrid Retrieval** — Combine dense CLIP embeddings with sparse BM25 keyword search
- [ ] **Batch/Async API** — REST endpoint for programmatic batch querying

---

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a PR, and ensure `pytest` passes on all changes.

## License

Released under the [MIT License](LICENSE).

---

<div align="center">

*Built with CLIP, FAISS, and Streamlit*

</div>