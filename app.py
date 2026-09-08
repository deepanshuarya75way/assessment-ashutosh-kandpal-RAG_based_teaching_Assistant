"""
Streamlit UI: ask where a topic appears in your course videos (RAG over Whisper chunks + bge-m3 + Ollama).
Run: streamlit run app.py
Prerequisites: Ollama (bge-m3 + a chat model), embeddings.joblib from preprocess_json.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TOP_K,
    format_timestamp,
    load_embeddings,
    rag_answer,
    retrieve_chunks,
)

DEFAULT_INDEX = ROOT / "embeddings.joblib"


def _css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; max-width: 960px; }
        div[data-testid="stExpander"] details summary { font-weight: 600; }
        .match-card {
            border-left: 4px solid #2563eb;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            background: rgba(37, 99, 235, 0.06);
            border-radius: 0 8px 8px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _load_df(path_str: str):
    return load_embeddings(path_str)


def main() -> None:
    st.set_page_config(
        page_title="Course Teaching Assistant",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    _css()

    st.title("Teaching assistant (RAG)")
    st.caption(
        "Video → Whisper segments → **bge-m3** vectors → cosine retrieval → **Ollama** answer "
        "with timestamps."
    )

    with st.sidebar:
        st.header("Settings")
        index_path = st.text_input("Embeddings file", value=str(DEFAULT_INDEX))
        ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)
        embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
        gen_model = st.text_input("Chat model", value=DEFAULT_GEN_MODEL)
        top_k = st.slider("Chunks to retrieve", min_value=3, max_value=20, value=DEFAULT_TOP_K)
        st.divider()
        st.markdown(
            "**Setup:** `ollama pull bge-m3` · `ollama pull llama3.2` · run `preprocess_json.py` after adding JSON in `jsons/`."
        )

    try:
        df = _load_df(index_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.success(f"Loaded **{len(df)}** transcript chunks.")

    q = st.text_area(
        "Your question",
        placeholder='e.g. "Where is CSS box model explained?" or "When does he talk about semantic HTML?"',
        height=100,
    )
    run = st.button("Find in videos & answer", type="primary", use_container_width=True)

    if run and q.strip():
        with st.spinner("Embedding query and retrieving matches…"):
            try:
                retrieved = retrieve_chunks(
                    q.strip(), df, ollama_url, embed_model, top_k
                )
            except Exception as e:
                st.error(f"Retrieval failed (is Ollama running?): {e}")
                st.stop()

        st.subheader("Where in the videos (retrieved context)")
        tbl = []
        for _, row in retrieved.iterrows():
            tbl.append(
                {
                    "Lesson": f"{row.get('number', '')} · {row.get('title', '')}",
                    "Time": f"{format_timestamp(row['start'])} – {format_timestamp(row['end'])}",
                    "Score": f"{float(row['_similarity']):.3f}",
                    "Excerpt": (str(row.get("text", ""))[:280] + "…")
                    if len(str(row.get("text", ""))) > 280
                    else str(row.get("text", "")),
                }
            )
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

        with st.expander("Full chunk text & metadata", expanded=False):
            for _, row in retrieved.iterrows():
                src = row.get("source_file", "")
                meta = f"**{row.get('title', '')}** · {format_timestamp(row['start'])}–{format_timestamp(row['end'])} · sim={float(row['_similarity']):.3f}"
                if pd.notna(src) and str(src):
                    meta += f" · `{src}`"
                st.markdown(meta)
                st.markdown(f"> {row.get('text', '')}")
                st.divider()

        with st.spinner("Generating answer with Ollama…"):
            try:
                answer, _ = rag_answer(
                    q.strip(),
                    df,
                    ollama_url,
                    embed_model,
                    gen_model,
                    top_k,
                    retrieved=retrieved,
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

        st.subheader("Answer")
        st.markdown(answer)

    elif run:
        st.warning("Enter a question first.")


if __name__ == "__main__":
    main()
