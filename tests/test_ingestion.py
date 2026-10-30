import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except ImportError:
    pass

try:
    import faiss
except ImportError:
    pass

import pytest
from pathlib import Path
from app.config import DATA_DIR
from app.ingestion.loaders import load_documents_from_directory
from app.ingestion.chunker import FinancialDocumentChunker
from app.ingestion.indexer import HybridIndexer


def test_load_documents_heterogeneous():
    documents = load_documents_from_directory(DATA_DIR)
    assert len(documents) > 0, "Should load at least one document from data directory"

    doc_types = {doc.document_type for doc in documents}
    # Must contain pdf, docx, txt, csv
    assert "pdf" in doc_types, "PDF documents must be loaded"
    assert "docx" in doc_types, "DOCX documents must be loaded"
    assert "txt" in doc_types, "TXT documents must be loaded"
    assert "csv" in doc_types, "CSV documents must be loaded"

    # Verify metadata fields
    for doc in documents:
        assert doc.document_name, "Document name must not be empty"
        assert doc.page_number >= 1, "Page number must be >= 1"
        assert len(doc.content) > 0, "Document content must not be empty"


def test_chunker_metadata_and_sizes():
    documents = load_documents_from_directory(DATA_DIR)
    chunker = FinancialDocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)

    assert len(chunks) > 0, "Chunks must be generated"

    for chunk in chunks:
        # Verify metadata schema
        meta = chunk.metadata
        assert isinstance(meta.document_name, str)
        assert isinstance(meta.document_type, str)
        assert isinstance(meta.page_number, int)
        assert isinstance(meta.chunk_id, str)
        assert chunk.chunk_id == meta.chunk_id
        assert chunk.chunk_id.startswith("chunk_")
        # Ensure text chunk is not empty
        assert len(chunk.content.strip()) > 0


def test_hybrid_indexer_build_and_search(tmp_path):
    documents = load_documents_from_directory(DATA_DIR)
    chunker = FinancialDocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)

    indexer = HybridIndexer(index_dir=tmp_path)
    indexer.build_indices(chunks)

    assert indexer.faiss_index is not None
    assert indexer.bm25 is not None
    assert indexer.faiss_index.ntotal == len(chunks)

    # Test search dense
    dense_res = indexer.search_dense("Nimbus Robotics revenue Q2 2026", top_k=5)
    assert len(dense_res) > 0
    assert dense_res[0][1] >= 0.0

    # Test search sparse
    sparse_res = indexer.search_sparse("debt-to-equity ratio revolver", top_k=5)
    assert len(sparse_res) > 0

    # Test save and reload
    indexer.save_indices()
    new_indexer = HybridIndexer(index_dir=tmp_path)
    loaded = new_indexer.load_indices()
    assert loaded is True
    assert len(new_indexer.chunks) == len(chunks)
