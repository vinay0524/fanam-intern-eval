from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.models import DocumentChunk, ChunkMetadata
from app.ingestion.loaders import LoadedDocument


class FinancialDocumentChunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: List[LoadedDocument]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_counter = 1

        for doc in documents:
            text_splits = self.text_splitter.split_text(doc.content)
            for split_idx, split_text in enumerate(text_splits):
                cleaned_text = split_text.strip()
                if not cleaned_text:
                    continue

                chunk_id = f"chunk_{chunk_counter}"
                chunk_counter += 1

                metadata = ChunkMetadata(
                    document_name=doc.document_name,
                    document_type=doc.document_type,
                    page_number=doc.page_number,
                    chunk_id=chunk_id,
                )

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=cleaned_text,
                    metadata=metadata,
                )
                chunks.append(chunk)

        return chunks

