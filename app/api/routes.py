from pathlib import Path
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.api.schemas import (
    FinanceQueryRequest,
    RetrieveRequest,
    IngestRequest,
    ClassifyRequest,
    RewriteRequest,
    DecomposeRequest
)

from app.services.finance_agent import (
    run_finance_agent,
    retrieve_context
)

from app.memory.session_store import (
    load_history,
    delete_history
)

from app.agent.classifier import classify_query
from app.agent.rewriter import rewrite_query
from app.agent.decomposer import decompose_query
from app.ingestion.index_builder import build_indexes

router = APIRouter()

RAW_DIR = Path("data/raw")

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".csv", ".xlsx"
}


@router.get("/")
def home():
    return {
        "project": "Adaptive Agentic RAG Financial Research Assistant",
        "status": "running"
    }


@router.get("/health")
def health():
    index_file = Path("data/indexes/faiss.index")
    bm25_file = Path("data/indexes/bm25.pkl")

    return {
        "status": "healthy",
        "faiss_index": "loaded" if index_file.exists() else "missing",
        "bm25_index": "loaded" if bm25_file.exists() else "missing"
    }


@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )

    destination = RAW_DIR / original_name

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="A file with this name already exists."
        )

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "message": "Document uploaded successfully",
            "filename": original_name,
            "status": "pending_ingestion"
        }

    except Exception:
        if destination.exists():
            destination.unlink()
        raise HTTPException(
            status_code=500,
            detail="Document upload failed."
        )

    finally:
        file.file.close()


@router.post("/documents/ingest")
def ingest_documents(request: IngestRequest):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        file for file in RAW_DIR.iterdir()
        if file.is_file()
        and file.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No supported documents found in data/raw."
        )

    try:
        # Rebuild chunks and both search indexes from raw documents.
        from scripts.ingest_documents import main as ingest_raw_documents

        ingest_raw_documents()
        build_indexes()

        return {
            "message": "Document ingestion completed",
            "documents_processed": len(files),
            "status": "success"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(error)}"
        )


@router.get("/documents")
def list_documents():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for file in RAW_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in ALLOWED_EXTENSIONS:
            documents.append({
                "filename": file.name,
                "file_type": file.suffix.lower().replace(".", ""),
                "size_bytes": file.stat().st_size
            })

    return {"documents": documents}


@router.get("/documents/{filename}")
def get_document_details(filename: str):
    file_path = RAW_DIR / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found."
        )

    return {
        "filename": file_path.name,
        "file_type": file_path.suffix.lower().replace(".", ""),
        "size_bytes": file_path.stat().st_size,
        "path": str(file_path)
    }


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    file_path = RAW_DIR / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found."
        )

    try:
        file_path.unlink()
        return {
            "message": f"Document '{filename}' deleted successfully.",
            "filename": filename
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(error)}"
        )


@router.post("/agent/classify")
def test_classify(request: ClassifyRequest):
    query_type = classify_query(request.query)
    return {
        "query": request.query,
        "query_type": query_type
    }


@router.post("/agent/rewrite")
def test_rewrite(request: RewriteRequest):
    rewritten = rewrite_query(request.query, history=request.history)
    return {
        "original_query": request.query,
        "rewritten_query": rewritten
    }


@router.post("/agent/decompose")
def test_decompose(request: DecomposeRequest):
    subqueries = decompose_query(request.query)
    return {
        "query": request.query,
        "subqueries": subqueries
    }


@router.post("/retrieve")
def retrieve_documents(request: RetrieveRequest):
    try:
        results = retrieve_context(
            request.query,
            request.top_k
        )

        return {
            "query": request.query,
            "total_results": len(results),
            "results": results
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(error)}"
        )


@router.post("/finance-agent")
def finance_agent(request: FinanceQueryRequest):
    try:
        return run_finance_agent(
            query=request.query,
            session_id=request.session_id,
            persona=request.persona,
            deep_search=request.deep_search
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Finance agent failed: {str(error)}"
        )


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    history = load_history(session_id)

    return {
        "session_id": session_id,
        "total_messages": len(history),
        "history": history
    }


@router.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    deleted = delete_history(session_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    return {
        "message": "Session history deleted",
        "session_id": session_id
    }
