"""
JSON transcript chunks -> Ollama bge-m3 embeddings -> pandas DataFrame -> joblib.
Requires: Ollama running with `ollama pull bge-m3`
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))


def create_embeddings(text_list: list[str]) -> list[list[float]]:
    all_emb: list[list[float]] = []
    for i in range(0, len(text_list), BATCH_SIZE):
        batch = text_list[i : i + BATCH_SIZE]
        r = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        if "embeddings" not in data:
            raise RuntimeError(f"Unexpected response: {data}")
        all_emb.extend(data["embeddings"])
    return all_emb


def main() -> None:
    json_dir = "jsons"
    out_path = "embeddings.joblib"
    os.makedirs(json_dir, exist_ok=True)

    jsons = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    if not jsons:
        raise SystemExit(f"No JSON files in ./{json_dir}/ — run mp3_to_json.py first.")

    my_dicts: list[dict] = []
    chunk_id = 0

    for json_file in sorted(jsons):
        path = os.path.join(json_dir, json_file)
        with open(path, encoding="utf-8") as f:
            content = json.load(f)
        chunks = content.get("chunks") or []
        if not chunks:
            print(f"Skip (no chunks): {json_file}")
            continue
        texts = [c["text"] for c in chunks]
        print(f"Embedding {len(texts)} segments from {json_file} …")
        embeddings = create_embeddings(texts)

        for i, chunk in enumerate(chunks):
            row = dict(chunk)
            row["chunk_id"] = chunk_id
            row["embedding"] = embeddings[i]
            row["source_file"] = json_file
            chunk_id += 1
            my_dicts.append(row)

    df = pd.DataFrame.from_records(my_dicts)
    joblib.dump(df, out_path)
    print(f"Saved {len(df)} chunks -> {out_path}")
    # Smoke check
    if len(df):
        m = np.vstack(df["embedding"].values)
        sim = cosine_similarity(m[:1], m[:2])[0, 1]
        print(f"Sample cosine(chunk0, chunk1) = {sim:.4f}")


if __name__ == "__main__":
    main()
