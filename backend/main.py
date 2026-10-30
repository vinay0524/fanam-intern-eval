import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from services.document_parser import DocumentParser
from services.text_chunker import TextChunker
from services.dense_service import DenseEmbeddingService
from services.sparse_service import SparseBM25Service
from services.pinecone_service import PineconeService
from services.llm_service import LLMService

app = FastAPI(
    title="Financial RAG with Hybrid Retrieval (Pinecone + Google Gemini)",
    description="Multi-format document intelligence (PDF, DOCX, CSV, TXT, XLSX) with Dense (Gemini) + Sparse (BM25) Retrieval",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service instances
text_chunker = TextChunker(chunk_size=700, chunk_overlap=150)
dense_service = DenseEmbeddingService()
sparse_service = SparseBM25Service()
pinecone_service = PineconeService()
llm_service = LLMService()

SUPPORTED_EXTENSIONS = {"docx", "pdf", "csv", "txt", "md", "xlsx", "xls"}

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query / question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top chunks to retrieve")
    alpha: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid weighting factor: 1.0 = Pure Dense, 0.0 = Pure Sparse (BM25), 0.5 = Balanced Hybrid"
    )

class ChunkMetadata(BaseModel):
    id: str
    score: float
    text: str
    source: str
    chunk_index: int

class QueryResponse(BaseModel):
    query: str
    answer: str
    alpha: float
    model: str
    sources: List[ChunkMetadata]

@app.get("/health")
@app.get("/api/health")
def health_check():
    key_status = settings.validate_keys()
    return {
        "status": "online",
        "api_keys_ready": key_status["configured"],
        "missing_keys": key_status["missing_keys"],
        "pinecone_index": settings.PINECONE_INDEX_NAME,
        "embedding_model": settings.GEMINI_EMBEDDING_MODEL,
        "llm_model": settings.GEMINI_LLM_MODEL,
        "default_alpha": settings.DEFAULT_ALPHA,
        "supported_formats": list(SUPPORTED_EXTENSIONS)
    }

@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...)
):
    """
    Multi-file Document Ingestion:
    Accepts one or multiple files simultaneously (.docx, .pdf, .csv, .txt, .xlsx).
    Parses, chunks, embeds with Gemini + BM25, and indexes into Pinecone.
    """
    validation = settings.validate_keys()
    if not validation["configured"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing API keys in backend/.env: {', '.join(validation['missing_keys'])}. Please configure them first."
        )

    processed_files = []
    total_chunks_all = 0
    total_upserted_all = 0

    for file in files:
        ext = file.filename.lower().split(".")[-1]
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        try:
            file_bytes = await file.read()
            if not file_bytes:
                continue

            # 1. Parse document
            extracted_text = DocumentParser.parse_file(file_bytes, filename=file.filename)
            if not extracted_text.strip():
                continue

            # 2. Text Chunking
            chunks = text_chunker.chunk_text(extracted_text, source_filename=file.filename)
            if not chunks:
                continue

            chunk_texts = [c["text"] for c in chunks]

            # 3. Dense Retrieval Vectors (Gemini embeddings)
            dense_vectors = dense_service.embed_texts(chunk_texts)

            # 4. Sparse Retrieval Vectors (BM25 encoding)
            sparse_vectors = sparse_service.encode_documents(chunk_texts)

            # 5. Hybrid Upsert into Pinecone
            upserted_count = pinecone_service.upsert_hybrid_chunks(
                chunks=chunks,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors
            )

            total_chunks_all += len(chunks)
            total_upserted_all += upserted_count

            processed_files.append({
                "filename": file.filename,
                "file_type": ext.upper(),
                "file_size": len(file_bytes),
                "total_characters": len(extracted_text),
                "chunks_created": len(chunks),
                "upserted_to_pinecone": upserted_count,
                "sample_text": extracted_text[:450].strip() + ("..." if len(extracted_text) > 450 else "")
            })

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")

    if not processed_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No supported or readable files could be processed."
        )

    return {
        "status": "success",
        "processed_files_count": len(processed_files),
        "files": processed_files,
        "total_chunks": total_chunks_all,
        "total_upserted": total_upserted_all,
        "message": f"Successfully indexed {len(processed_files)} document(s) with {total_chunks_all} hybrid chunks into Pinecone."
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Hybrid Search Query:
    1. Dense query vector (Google Gemini)
    2. Sparse query vector (BM25)
    3. Pinecone Hybrid search (scaled by alpha)
    4. Google Gemini LLM synthesis
    """
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    validation = settings.validate_keys()
    if not validation["configured"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing API keys in backend/.env: {', '.join(validation['missing_keys'])}. Please configure them first."
        )

    try:
        alpha = request.alpha if request.alpha is not None else settings.DEFAULT_ALPHA

        # 1. Generate Dense Query Vector (Gemini)
        dense_query_vec = dense_service.embed_query(request.query)

        # 2. Generate Sparse Query Vector (BM25)
        sparse_query_vec = sparse_service.encode_query(request.query)

        # 3. Perform Hybrid Search on Pinecone
        retrieved_chunks = pinecone_service.hybrid_query(
            dense_vector=dense_query_vec,
            sparse_vector=sparse_query_vec,
            top_k=request.top_k,
            alpha=alpha
        )

        if not retrieved_chunks:
            return QueryResponse(
                query=request.query,
                answer="No relevant documents or chunks were found in the Pinecone vector index. Please upload a document first.",
                alpha=alpha,
                model=settings.GEMINI_LLM_MODEL,
                sources=[]
            )

        # 4. Synthesize final answer via Gemini LLM
        llm_result = llm_service.generate_answer(
            query=request.query,
            context_chunks=retrieved_chunks
        )

        sources = [
            ChunkMetadata(
                id=c["id"],
                score=c["score"],
                text=c["text"],
                source=c["source"],
                chunk_index=c["chunk_index"]
            )
            for c in retrieved_chunks
        ]

        return QueryResponse(
            query=request.query,
            answer=llm_result["answer"],
            alpha=alpha,
            model=llm_result["model"],
            sources=sources
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
