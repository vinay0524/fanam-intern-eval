
import json
from pathlib import Path

import faiss

from app.ingestion.embeddings import generate_embeddings

INDEX_DIR = Path("data/indexes")
FAISS_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.json"


def dense_search(query: str, top_k: int = 5) -> list:
    if not FAISS_FILE.exists() or not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "FAISS index or chunks.json not found. Build indexes first."
        )

    index = faiss.read_index(str(FAISS_FILE))

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        return []

    query_embedding = generate_embeddings([query])

    k = min(top_k, index.ntotal)

    if k == 0:
        return []

    scores, positions = index.search(query_embedding, k)

    results = []

    for score, position in zip(scores[0], positions[0]):
        if position < 0 or position >= len(chunks):
            continue

        chunk = chunks[position]

        results.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "score": float(score)
        })

    return results
