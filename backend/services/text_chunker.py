import re
from typing import List, Dict, Any

class TextChunker:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 150):
        """
        Pure Python Text Chunker for Financial & Multi-Format Documents.
        Ensures table integrity, preserves sentence boundaries, and creates
        clean ASCII IDs for Pinecone compatibility.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source_filename: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        # Sanitize filename for Pinecone ID
        safe_prefix = re.sub(r"[^a-zA-Z0-9_\-]", "_", source_filename)

        cleaned_text = re.sub(r"\r\n", "\n", text).strip()
        
        # Split on paragraph or table boundaries
        paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk_parts = []
        current_length = 0
        chunk_index = 0

        for para in paragraphs:
            para_len = len(para)

            # If adding this paragraph exceeds chunk_size and we already have parts
            if current_length + para_len > self.chunk_size and current_chunk_parts:
                chunk_str = "\n\n".join(current_chunk_parts).strip()
                chunks.append({
                    "chunk_id": f"{safe_prefix}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "text": chunk_str,
                    "source": source_filename,
                    "char_count": len(chunk_str)
                })
                chunk_index += 1

                # Retain overlap from previous chunk
                overlap_chars = 0
                new_parts = []
                for part in reversed(current_chunk_parts):
                    if overlap_chars + len(part) <= self.chunk_overlap:
                        new_parts.insert(0, part)
                        overlap_chars += len(part)
                    else:
                        break
                current_chunk_parts = new_parts
                current_length = sum(len(p) for p in current_chunk_parts)

            # If single paragraph exceeds chunk size (e.g. large table or long text), split by lines or sentences
            if para_len > self.chunk_size:
                # If it's a table with multiple lines, split by lines first
                if "\n" in para:
                    sub_units = [line.strip() for line in para.split("\n") if line.strip()]
                else:
                    sub_units = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]

                for unit in sub_units:
                    if current_length + len(unit) > self.chunk_size and current_chunk_parts:
                        chunk_str = "\n".join(current_chunk_parts).strip()
                        chunks.append({
                            "chunk_id": f"{safe_prefix}_chunk_{chunk_index}",
                            "chunk_index": chunk_index,
                            "text": chunk_str,
                            "source": source_filename,
                            "char_count": len(chunk_str)
                        })
                        chunk_index += 1
                        current_chunk_parts = []
                        current_length = 0
                    current_chunk_parts.append(unit)
                    current_length += len(unit)
            else:
                current_chunk_parts.append(para)
                current_length += para_len

        # Append trailing chunk
        if current_chunk_parts:
            chunk_str = "\n\n".join(current_chunk_parts).strip()
            if chunk_str:
                chunks.append({
                    "chunk_id": f"{safe_prefix}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "text": chunk_str,
                    "source": source_filename,
                    "char_count": len(chunk_str)
                })

        return chunks
