"""
CLI RAG Q&A (same logic as Streamlit). Writes prompt.txt and response.txt for debugging.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_GEN_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TOP_K,
    build_rag_prompt,
    format_timestamp,
    load_embeddings,
    generate_answer,
    retrieve_chunks,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Ask the course RAG assistant (Ollama + bge-m3).")
    p.add_argument("question", nargs="?", help="Question (omit to type interactively)")
    p.add_argument("--index", default="embeddings.joblib", help="Path to joblib index")
    p.add_argument("--ollama", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--no-llm", action="store_true", help="Only print retrieved chunks, skip generate")
    args = p.parse_args()

    q = args.question or input("Ask a question: ").strip()
    if not q:
        print("Empty question.", file=sys.stderr)
        sys.exit(1)

    df = load_embeddings(Path(args.index))
    retrieved = retrieve_chunks(q, df, args.ollama, args.embed_model, args.top_k)

    print("\n--- Top matches (where in the videos) ---\n")
    for _, row in retrieved.iterrows():
        t0, t1 = format_timestamp(row["start"]), format_timestamp(row["end"])
        print(
            f"[{float(row['_similarity']):.3f}] #{row.get('number', '')} {row.get('title', '')} "
            f"· {t0}–{t1}"
        )
        print(f"    {str(row.get('text', '')).strip()[:500]}{'…' if len(str(row.get('text', ''))) > 500 else ''}\n")
    if args.no_llm:
        return

    prompt = build_rag_prompt(q, retrieved)
    (ROOT / "prompt.txt").write_text(prompt, encoding="utf-8")
    answer = generate_answer(prompt, args.ollama, args.gen_model)
    (ROOT / "response.txt").write_text(answer, encoding="utf-8")
    print("--- Answer ---\n")
    print(answer)


if __name__ == "__main__":
    main()
