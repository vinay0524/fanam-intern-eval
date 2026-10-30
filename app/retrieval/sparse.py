
import pickle
from pathlib import Path


INDEX_DIR = Path("data/indexes")
BM25_FILE = INDEX_DIR / "bm25.pkl"
DOCUMENTS_FILE = INDEX_DIR / "bm25_documents.pkl"


def sparse_search(query: str, top_k: int = 5) -> list:
    if not BM25_FILE.exists() or not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            "BM25 index files not found. Build indexes first."
        )

    with open(BM25_FILE, "rb") as file:
        bm25 = pickle.load(file)

    with open(DOCUMENTS_FILE, "rb") as file:
        documents = pickle.load(file)

    if not documents:
        return []

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    ranked_positions = sorted(
        range(len(scores)),
        key=lambda position: scores[position],
        reverse=True
    )

    results = []

    for position in ranked_positions[:top_k]:
        score = float(scores[position])

        if score <= 0:
            continue

        chunk = documents[position]

        results.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "score": score
        })

    return results
