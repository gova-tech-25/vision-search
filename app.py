import os
import streamlit as st
from PIL import Image
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="CLIP Image Retrieval Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphic Theme)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-title {
        text-align: center;
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: rgba(31, 41, 55, 0.6);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
    }
    .card-rank {
        position: absolute;
        top: 10px;
        left: 10px;
        background: rgba(79, 70, 229, 0.9);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .score-badge {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #34D399;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# Lazy Cache Resource for RetrievalEngine
@st.cache_resource(show_spinner="Loading CLIP Model & Image Index...")
def load_engine():
    from src.retrieval import RetrievalEngine
    return RetrievalEngine()

def main():
    st.markdown('<div class="main-title">CLIP Text-to-Image Retrieval Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Powered by OpenAI CLIP & Flickr30k Multimodal Embeddings</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Search Configuration")

    top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=20, value=6, step=1)

    backend_options = {
        "Exact — Brute Force": "bruteforce",
        "Exact — FAISS FlatIP": "flat",
        "Fast — FAISS HNSW (Approximate)": "hnsw",
    }
    backend_label = st.sidebar.selectbox("Vector Search Backend", list(backend_options.keys()), index=0)
    backend = backend_options[backend_label]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Information")

    # Check if index exists
    from src.config import EMBEDDINGS_PATH, METADATA_PATH, HNSW_INDEX_PATH, MODEL_NAME, DEVICE
    index_exists = EMBEDDINGS_PATH.exists() and METADATA_PATH.exists()

    st.sidebar.text(f"Model: {MODEL_NAME}")
    st.sidebar.text(f"Device: {DEVICE.upper()}")
    st.sidebar.text(f"FlatIP Index: {'LOADED' if index_exists else 'MISSING'}")
    st.sidebar.text(f"HNSW Index: {'LOADED' if HNSW_INDEX_PATH.exists() else 'MISSING'}")

    if not index_exists:
        st.error("""
            ### ⚠️ Image Index Not Found!
            The precomputed vector index has not been built yet.

            **Please run the indexing pipeline in your terminal:**
            ```bash
            python -m src.index
            ```
            Once indexing completes, refresh this page.
        """)
        return

    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"Error loading retrieval engine: {e}")
        return

    st.sidebar.text(f"Total Indexed Images: {engine.num_images:,}")

    # Query Section
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        query = st.text_input(
            "Enter your natural language search query:",
            placeholder="e.g. A dog running through a grassy field",
            key="query_input"
        )

    # Example Query Chips
    st.markdown("**Try example queries:**")
    example_queries = [
        "A dog running through a grassy field",
        "A child playing in a wooden playhouse",
        "A group of people sitting at a cafe table",
        "A person skiing down a snowy mountain slope",
    ]

    cols_examples = st.columns(len(example_queries))

    def _set_query(example: str):
        # on_click callbacks run before widgets are instantiated, so this
        # doesn't trip Streamlit's "cannot modify after instantiation" guard
        st.session_state["query_input"] = example

    for idx, ex in enumerate(example_queries):
        cols_examples[idx].button(ex, key=f"ex_{idx}", on_click=_set_query, args=(ex,))

    if not query or not query.strip():
        st.info("👆 Enter a natural language description above to retrieve matching images.")
        return

    # Execute Retrieval
    with st.spinner("Searching multimodal vector index..."):
        try:
            search_results = engine.search(query=query, k=top_k, backend=backend)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return

    # Display Metrics Bar
    latency = search_results["latency"]
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{latency["text_encoding_ms"]} ms</div><div class="metric-label">Text Encoding Latency</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{latency["search_ms"]} ms</div><div class="metric-label">Vector Search Latency ({backend.upper()})</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{latency["total_ms"]} ms</div><div class="metric-label">Total Latency</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{len(search_results["results"])}</div><div class="metric-label">Retrieved Images</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"Retrieved Top-{top_k} Results for: *\"{query}\"*")

    # Display Image Results Grid
    results = search_results["results"]

    if not results:
        st.warning("No results found for this query.")
        return

    # 3 columns layout
    grid_cols = st.columns(min(3, len(results)))

    for idx, item in enumerate(results):
        col = grid_cols[idx % len(grid_cols)]

        with col:
            img_path = Path(item["image_path"])

            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error loading image: {e}")
            else:
                st.error(f"Image file not found: {item['image_id']}")

            st.markdown(f"**Rank {item['rank']}** | ID: `{item['image_id']}`")
            st.markdown(f'<div class="score-badge">Cosine Similarity: {item["score"]:.4f}</div>', unsafe_allow_html=True)

            if item.get("captions"):
                with st.expander("Ground Truth Captions"):
                    for cap in item["captions"]:
                        st.markdown(f"- {cap}")

            st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
