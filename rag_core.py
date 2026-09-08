"""
RAG retrieval: query embedding (Ollama bge-m3) + cosine similarity (sklearn).
Generation: Ollama /api/generate with grounded prompt including timestamps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_GEN_MODEL = "llama3.2"
DEFAULT_TOP_K = 8


def format_timestamp(seconds: float) -> str:
    """MM:SS for display; supports lessons longer than an hour."""
    s = float(seconds)
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def load_embeddings(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings index not found: {path}")
    df = joblib.load(path)
    if "embedding" not in df.columns:
        raise ValueError("DataFrame must contain an 'embedding' column")
    return df


def create_embeddings(
    texts: list[str],
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
    timeout: int = 300,
) -> list[list[float]]:
    """Call Ollama /api/embed; batch to avoid huge single payloads."""
    base = base_url.rstrip("/")
    all_emb: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        r = requests.post(
            f"{base}/api/embed",
            json={"model": model, "input": batch},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if "embeddings" not in data:
            raise RuntimeError(f"Unexpected embed response: {data}")
        all_emb.extend(data["embeddings"])
    return all_emb


def retrieve_chunks(
    query: str,
    df: pd.DataFrame,
    base_url: str = DEFAULT_OLLAMA_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Embed query and return top-k rows by cosine similarity."""
    if not query.strip():
        return df.iloc[0:0].copy()
    q_emb = create_embeddings([query.strip()], base_url, embed_model, batch_size=1)[0]
    mat = np.vstack(df["embedding"].values)
    sims = cosine_similarity(mat, np.asarray(q_emb, dtype=np.float64).reshape(1, -1)).flatten()
    k = min(top_k, len(df))
    top_idx = sims.argsort()[::-1][:k]
    out = df.iloc[top_idx].copy()
    out["_similarity"] = sims[top_idx]
    return out


def build_rag_prompt(query: str, retrieved: pd.DataFrame) -> str:
    rows: list[dict[str, Any]] = []
    for _, row in retrieved.iterrows():
        item = {
            "video_title": row.get("title", ""),
            "lesson_number": str(row.get("number", "")),
            "start_sec": float(row["start"]),
            "end_sec": float(row["end"]),
            "start_time": format_timestamp(row["start"]),
            "end_time": format_timestamp(row["end"]),
            "text": str(row.get("text", "")).strip(),
            "similarity": round(float(row["_similarity"]), 4),
        }
        if "source_file" in row and pd.notna(row["source_file"]):
            item["source_json"] = str(row["source_file"])
        rows.append(item)
    ctx = json.dumps(rows, indent=2, ensure_ascii=False)
    return f"""You are a teaching assistant for a video course. Answer using ONLY the context below.

When the student asks where a topic appears, always give:
- lesson title and number,
- start and end timestamps (use the start_time / end_time fields),
- a short quote or paraphrase from the text as evidence.

If the context is insufficient, say so and suggest what keyword they might search for.

Context (retrieved transcript chunks):
{ctx}

Student question:
{query}

Answer clearly. Use bullet points when listing multiple locations in videos."""


def generate_answer(
    prompt: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_GEN_MODEL,
    timeout: int = 600,
) -> str:
    base = base_url.rstrip("/")
    r = requests.post(
        f"{base}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return str(data.get("response", ""))


def rag_answer(
    query: str,
    df: pd.DataFrame,
    base_url: str = DEFAULT_OLLAMA_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    gen_model: str = DEFAULT_GEN_MODEL,
    top_k: int = DEFAULT_TOP_K,
    retrieved: pd.DataFrame | None = None,
) -> tuple[str, pd.DataFrame]:
    """Retrieve + generate; returns (answer_text, retrieved_df). Pass `retrieved` to skip a second embed call."""
    r = (
        retrieved
        if retrieved is not None
        else retrieve_chunks(query, df, base_url, embed_model, top_k)
    )
    if r.empty:
        return (
            "No indexed content loaded. Build embeddings with preprocess_json.py first.",
            r,
        )
    prompt = build_rag_prompt(query, r)
    answer = generate_answer(prompt, base_url, gen_model)
    return answer, r
