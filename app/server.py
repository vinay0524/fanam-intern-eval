import os
from pathlib import Path

# Ensure OpenMP compatibility on Windows between PyTorch and FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import torch
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.models import AgentRequest, AgentResponse
from app.agent import FinancialResearchAgent

# Global agent instance
agent = FinancialResearchAgent()

STATIC_HTML_PATH = Path(__file__).parent / "static" / "index.html"

app = FastAPI(
    title="Financial Research Assistant Agentic RAG API",
    description="Production-Grade Multi-Persona Financial Research Assistant with Hybrid RRF Retrieval, Dynamic Query Decomposition, and Faithfulness Verification.",
    version="1.0.0",
)

# Enable CORS for frontend/dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    if not agent.is_initialized:
        agent.initialize()


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    """
    Serves the interactive Financial Research Assistant Dashboard.
    """
    if not STATIC_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found")
    return HTMLResponse(content=STATIC_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if not agent.is_initialized:
        agent.initialize()
    return {
        "status": "healthy",
        "agent_initialized": agent.is_initialized,
        "indexed_chunks": len(agent.indexer.chunks),
    }


@app.post("/finance-agent", response_model=AgentResponse, status_code=status.HTTP_200_OK)
def finance_agent_endpoint(request: AgentRequest):
    """
    Main endpoint for Financial Research Assistant Agentic RAG pipeline.
    """
    try:
        if not agent.is_initialized:
            agent.initialize()
        response = agent.run(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


@app.post("/ingest", status_code=status.HTTP_200_OK)
def trigger_ingestion(force_reindex: bool = True):
    """
    Trigger document re-indexing from the ./data directory.
    """
    try:
        agent.initialize(force_reindex=force_reindex)
        return {
            "status": "success",
            "message": "Document ingestion and hybrid indexing completed successfully.",
            "chunk_count": len(agent.indexer.chunks),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
