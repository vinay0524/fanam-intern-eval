from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Adaptive Agentic RAG Financial Research Assistant",
    description=(
        "A financial research assistant using "
        "Agentic RAG, Gemini, FAISS and BM25."
    ),
    version="1.0.0"
)

app.include_router(router)
