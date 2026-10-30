import os
import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Ensure OpenMP compatibility on Windows between PyTorch and FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
except ImportError:
    pass

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    STORAGE_DIR,
)
from app.models import DocumentChunk, ChunkMetadata


def financial_tokenizer(text: str) -> List[str]:
    """
    Financial-aware tokenizer preserving tickers, dollar values, percentages, and metrics.
    """
    text = text.lower()
    # Normalize unicode hyphens/quotes
    text = re.sub(r"[—–−]", "-", text)
    # Tokenize preserving words, numbers with decimals, percentages, and currencies
    tokens = re.findall(r"\b[a-z0-9]+(?:[\.\-][a-z0-9]+)*%?|\$[0-9]+(?:\.[0-9]+)?", text)
    return tokens


class HybridIndexer:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        index_dir: Path = INDEX_DIR,
    ):
        self.model_name = model_name
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[SentenceTransformer] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self.chunk_id_map: Dict[str, DocumentChunk] = {}

    def _get_embedding_model(self) -> SentenceTransformer:
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(self.model_name)
        return self.model

    def build_indices(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        self.chunk_id_map = {chunk.chunk_id: chunk for chunk in chunks}

        if not chunks:
            print("Warning: No chunks provided to index.")
            return

        corpus_texts = [chunk.content for chunk in chunks]

        # 1. Dense Index (FAISS with Cosine Similarity / Inner Product on Normalized Vectors)
        model = self._get_embedding_model()
        embeddings = model.encode(
            corpus_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings.astype(np.float32))

        # 2. Sparse Index (BM25Okapi)
        tokenized_corpus = [financial_tokenizer(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Successfully built Dense FAISS ({len(chunks)} vectors, dim={dimension}) and Sparse BM25 indices.")

    def save_indices(self) -> None:
        if self.faiss_index is None or self.bm25 is None or not self.chunks:
            raise ValueError("Indices are not built yet.")

        # Save FAISS index
        faiss_path = self.index_dir / "faiss.index"
        faiss.write_index(self.faiss_index, str(faiss_path))

        # Save BM25
        bm25_path = self.index_dir / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)

        # Save Chunks metadata and content
        chunks_path = self.index_dir / "chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            chunks_data = [chunk.model_dump() for chunk in self.chunks]
            json.dump(chunks_data, f, indent=2)

        print(f"Indices saved successfully to {self.index_dir}")

    def load_indices(self) -> bool:
        faiss_path = self.index_dir / "faiss.index"
        bm25_path = self.index_dir / "bm25.pkl"
        chunks_path = self.index_dir / "chunks.json"

        if not (faiss_path.exists() and bm25_path.exists() and chunks_path.exists()):
            return False

        try:
            self.faiss_index = faiss.read_index(str(faiss_path))

            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)

            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                self.chunks = [DocumentChunk(**item) for item in chunks_data]
                self.chunk_id_map = {chunk.chunk_id: chunk for chunk in self.chunks}

            # Warm embedding model
            self._get_embedding_model()
            print(f"Loaded existing indices with {len(self.chunks)} chunks from {self.index_dir}")
            return True
        except Exception as e:
            print(f"Failed to load existing indices: {e}")
            return False

    def search_dense(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if self.faiss_index is None or not self.chunks:
            return []
        model = self._get_embedding_model()
        query_vec = model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        k = min(top_k, len(self.chunks))
        scores, indices = self.faiss_index.search(query_vec, k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append((chunk, float(score)))
        return results

    def search_sparse(self, query: str, top_k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        if self.bm25 is None or not self.chunks:
            return []
        tokenized_query = financial_tokenizer(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            chunk = self.chunks[idx]
            results.append((chunk, score))
        return results
