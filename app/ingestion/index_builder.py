import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import pickle

import faiss
from rank_bm25 import BM25Okapi

from app.ingestion.embeddings import generate_embeddings


INDEX_DIR = Path("data/indexes")
CHUNKS_FILE = INDEX_DIR / "chunks.json"


def build_indexes():

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load document chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No document chunks found.")

    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    embeddings = generate_embeddings(texts)

    # Build FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        str(INDEX_DIR / "faiss.index")
    )

    # Build BM25 index
    tokenized_corpus = [
        text.lower().split() for text in texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    with open(INDEX_DIR / "bm25.pkl", "wb") as file:
        pickle.dump(bm25, file)

    # Save BM25 document mapping
    with open(INDEX_DIR / "bm25_documents.pkl", "wb") as file:
        pickle.dump(chunks, file)

    print("FAISS index created successfully.")
    print("BM25 index created successfully.")
    print(f"Total indexed chunks: {len(chunks)}")


if __name__ == "__main__":
    build_indexes()
