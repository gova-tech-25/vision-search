import os
from pathlib import Path
from PIL import Image
import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CLIP-Retrieve • Production Multimodal Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Premium Dark Glassmorphic Theme & Modern Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 1rem 1rem 1rem;
        background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.15) 0%, rgba(15, 23, 42, 0) 70%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }

    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin: 2px 4px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(30, 41, 59, 0.6);
        color: #CBD5E1;
    }

    .badge-accent {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
        color: #A5B4FC;
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #34D399;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
    }

    .metric-card-highlight {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(14, 165, 233, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .metric-value-green {
        color: #34D399;
    }

    .metric-value-purple {
        color: #C084FC;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .result-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 12px;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .result-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .score-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        margin-top: 4px;
    }

    .match-badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #A5B4FC;
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Lazy Engine Cache
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CLIP Model & FAISS Vectors...")
def get_retrieval_engine():
    from src.retrieval import RetrievalEngine
    return RetrievalEngine()


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Header Banner
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">🔍 CLIP-Retrieve</div>
            <div class="sub-title">Production-Grade Multimodal Text-to-Image Search on Flickr30k</div>
            <div>
                <span class="badge-pill badge-accent">OpenAI CLIP ViT-B/32</span>
                <span class="badge-pill">31,783 Corpus Images</span>
                <span class="badge-pill">512-D L2 Vectors</span>
                <span class="badge-pill badge-success">FAISS FlatIP & HNSW</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Sidebar Configuration & Telemetry
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Search Configuration")

        top_k = st.slider("Top-K Retrieved Items", min_value=1, max_value=20, value=6, step=1)

        backend_options = {
            "Fast — FAISS HNSW (Approximate)": "hnsw",
            "Exact — FAISS FlatIP (Ground Truth)": "flat",
            "Exact — Brute Force (NumPy Dot Product)": "bruteforce",
        }
        backend_label = st.selectbox(
            "Vector Search Backend",
            list(backend_options.keys()),
            index=0,
            help="HNSW uses Hierarchical Navigable Small World graphs for sub-millisecond search; FlatIP computes exact inner product.",
        )
        selected_backend = backend_options[backend_label]

        # Dynamic efSearch Slider for HNSW
        ef_search_val = 64
        if selected_backend == "hnsw":
            st.markdown("#### ⚡ HNSW Beam Width (`efSearch`)")
            ef_search_val = st.slider(
                "efSearch Parameter",
                min_value=16,
                max_value=256,
                value=64,
                step=8,
                help="Controls exploration beam width during HNSW search. Higher values yield higher recall approaching 100% at slight latency cost.",
            )

            if ef_search_val < 32:
                hint = "⚡ Ultra-Fast (~80-87% Recall@10, <0.25ms)"
            elif ef_search_val <= 96:
                hint = "🎯 Recommended Sweet Spot (~90-97% Recall@10, ~0.35ms)"
            else:
                hint = "🔬 High Precision (~98-100% Recall@10, ~0.6-1.5ms)"
            st.caption(hint)

        st.markdown("---")
        st.subheader("📊 System & Index Status")

        from src.config import (
            EMBEDDINGS_PATH,
            METADATA_PATH,
            FAISS_INDEX_PATH,
            HNSW_INDEX_PATH,
            MODEL_NAME,
            DEVICE,
            HNSW_M,
            HNSW_EF_CONSTRUCTION,
        )

        flat_exists = FAISS_INDEX_PATH.exists()
        hnsw_exists = HNSW_INDEX_PATH.exists()
        embeddings_exist = EMBEDDINGS_PATH.exists() and METADATA_PATH.exists()

        st.text(f"Model: {MODEL_NAME.split('/')[-1]}")
        st.text(f"Device: {DEVICE.upper()}")
        st.text(f"Corpus Embeddings: {'LOADED' if embeddings_exist else 'MISSING'}")
        st.text(f"FAISS FlatIP Index: {'LOADED' if flat_exists else 'MISSING'}")
        st.text(f"FAISS HNSW Index:   {'LOADED' if hnsw_exists else 'MISSING'}")
        st.caption(f"HNSW Params: M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}")

    if not embeddings_exist:
        st.error(
            """
            ### ⚠️ Vector Index Not Found
            Offline indexing has not been run. Please generate embeddings first:
            ```bash
            python -m src.index
            ```
            """
        )
        return

    try:
        engine = get_retrieval_engine()
    except Exception as exc:
        st.error(f"Failed to initialize RetrievalEngine: {exc}")
        return

    # -------------------------------------------------------------------------
    # Main Navigation Tabs
    # -------------------------------------------------------------------------
    tab1, tab2 = st.tabs(
        [
            "🔍 Live Image Search",
            "⚖️ FlatIP vs HNSW Comparison",
        ]
    )

    # =========================================================================
    # TAB 1: Live Image Search
    # =========================================================================
    with tab1:
        st.markdown("##### 📝 Natural Language Query")

        col_input, _ = st.columns([6, 1])
        with col_input:
            search_query = st.text_input(
                "Enter a description:",
                placeholder="e.g. A dog running through grass",
                key="tab1_query_input",
            )

        # Quick Example Chips
        st.caption("Try example queries:")
        example_queries = [
            "a dog running through grass",
            "three women sitting in a village",
            "children playing outside",
            "a man riding a bicycle",
            "a person skiing on snow",
        ]

        def _select_query(q):
            st.session_state["tab1_query_input"] = q

        cols_chips = st.columns(len(example_queries))
        for i, q in enumerate(example_queries):
            cols_chips[i].button(q, key=f"tab1_chip_{i}", on_click=_select_query, args=(q,))

        if search_query and search_query.strip():
            with st.spinner("Embedding query & searching vector index..."):
                try:
                    res = engine.search(
                        query=search_query,
                        k=top_k,
                        backend=selected_backend,
                        ef_search=ef_search_val if selected_backend == "hnsw" else None,
                    )
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    return

            lat = res["latency"]
            # Metrics Bar
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{lat['text_encoding_ms']} ms</div>
                        <div class="metric-label">Text Encoding</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="metric-card metric-card-highlight">
                        <div class="metric-value metric-value-purple">{lat['search_ms']} ms</div>
                        <div class="metric-label">{selected_backend.upper()} Search Time</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value metric-value-green">{lat['total_ms']} ms</div>
                        <div class="metric-label">Total Latency</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{len(res['results'])}</div>
                        <div class="metric-label">Top Results</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(f"#### Top-{top_k} Retrieved Results for: *\"{search_query}\"*")

            # Image Grid
            results = res["results"]
            if not results:
                st.warning("No matches found.")
            else:
                cols_grid = st.columns(min(3, len(results)))
                for idx, item in enumerate(results):
                    c = cols_grid[idx % len(cols_grid)]
                    with c:
                        img_path = Path(item["image_path"])
                        if img_path.exists():
                            try:
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            except Exception as img_err:
                                st.caption(f"Image error: {img_err}")
                        else:
                            st.caption(f"File missing: {item['image_id']}")

                        st.markdown(
                            f"**Rank {item['rank']}** • `{item['image_id']}`<br>"
                            f"<span class='score-badge'>Cosine Sim: {item['score']:.4f}</span>",
                            unsafe_allow_html=True,
                        )

                        if item.get("captions"):
                            with st.expander("Ground Truth Captions"):
                                for cap in item["captions"]:
                                    st.markdown(f"- {cap}")
                        st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("👆 Enter a natural-language description or click an example above to retrieve images.")

    # =========================================================================
    # TAB 2: FlatIP vs HNSW Side-by-Side Comparison
    # =========================================================================
    with tab2:
        st.markdown("### ⚖️ Side-by-Side Comparison: Exact FlatIP vs Fast HNSW")
        st.markdown(
            "Evaluates retrieval consistency and latency speedup between approximate HNSW "
            "and exact FlatIP ground truth on the identical query embedding."
        )

        cmp_col1, cmp_col2 = st.columns([5, 2])
        with cmp_col1:
            cmp_query = st.text_input(
                "Enter test query for comparative evaluation:",
                value="three women sitting in a village",
                key="tab2_cmp_query",
            )
        with cmp_col2:
            cmp_ef = st.slider(
                "HNSW efSearch Beam Width",
                min_value=16,
                max_value=256,
                value=64,
                step=16,
                key="tab2_ef",
            )

        if st.button("Run Side-by-Side Comparison", type="primary", key="btn_run_cmp"):
            with st.spinner("Executing simultaneous FlatIP and HNSW searches..."):
                cmp_res = engine.compare_backends(
                    query=cmp_query,
                    k=top_k,
                    backends=["flat", "hnsw"],
                    ef_search=cmp_ef,
                )

            comp = cmp_res.get("comparison", {})
            b_results = cmp_res.get("backend_results", {})
            flat_data = b_results.get("flat", {})
            hnsw_data = b_results.get("hnsw", {})

            # Summary KPI Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"""<div class="metric-card metric-card-highlight">
                        <div class="metric-value metric-value-green">{comp.get('recall_overlap_pct', 0.0)}%</div>
                        <div class="metric-label">Recall@{top_k} Overlap</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{comp.get('hnsw_speedup_factor', 1.0)}x</div>
                        <div class="metric-label">HNSW Vector Speedup</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c3:
                flat_ms = flat_data.get("latency", {}).get("search_ms", 0.0)
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{flat_ms} ms</div>
                        <div class="metric-label">FlatIP Vector Search</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with c4:
                hnsw_ms = hnsw_data.get("latency", {}).get("search_ms", 0.0)
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value metric-value-purple">{hnsw_ms} ms</div>
                        <div class="metric-label">HNSW Vector Search</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Dual Comparison Columns
            col_flat, col_hnsw = st.columns(2)
            flat_items = flat_data.get("results", [])
            hnsw_items = hnsw_data.get("results", [])
            exact_ids = set(r["image_id"] for r in flat_items)

            with col_flat:
                st.markdown(f"#### 🎯 FAISS FlatIP (Exact Ground Truth)")
                st.caption(f"Vectors searched: {engine.num_images:,} | Search: {flat_ms} ms")
                for item in flat_items:
                    st.markdown(
                        f"""<div class="result-card">
                            <strong>Rank {item['rank']}</strong> • <code>{item['image_id']}</code>
                            <span class="score-badge">Sim: {item['score']:.4f}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    p = Path(item["image_path"])
                    if p.exists():
                        st.image(Image.open(p), use_container_width=True)
                    if item.get("captions"):
                        st.caption(f"Caption: {item['captions'][0]}")
                    st.markdown("<br>", unsafe_allow_html=True)

            with col_hnsw:
                st.markdown(f"#### ⚡ FAISS HNSW (Approximate, efSearch={cmp_ef})")
                st.caption(f"Graph nodes: {engine.num_images:,} | Search: {hnsw_ms} ms")
                for item in hnsw_items:
                    is_match = item["image_id"] in exact_ids
                    match_badge = (
                        '<span class="match-badge">🎯 Exact Match</span>'
                        if is_match
                        else '<span class="badge-pill">Novel</span>'
                    )
                    st.markdown(
                        f"""<div class="result-card">
                            <strong>Rank {item['rank']}</strong> • <code>{item['image_id']}</code>
                            <span class="score-badge">Sim: {item['score']:.4f}</span>
                            {match_badge}
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    p = Path(item["image_path"])
                    if p.exists():
                        st.image(Image.open(p), use_container_width=True)
                    if item.get("captions"):
                        st.caption(f"Caption: {item['captions'][0]}")
                    st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
