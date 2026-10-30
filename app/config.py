import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_DIR = STORAGE_DIR / "indices"

# Environment Variable & .env Support
ENV_FILE = BASE_DIR / ".env"
if not ENV_FILE.exists():
    try:
        ENV_FILE.write_text("GEMINI_API_KEY=\nOPENAI_API_KEY=\n", encoding="utf-8")
    except Exception:
        pass

load_dotenv(dotenv_path=ENV_FILE)

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

# Model and Chunking configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval configurations
RRF_K = 60
TOP_K_DENSE = 10
TOP_K_SPARSE = 10
FINAL_TOP_K = 6
SUFFICIENCY_SIMILARITY_THRESHOLD = 0.30
MIN_SUFFICIENCY_CHUNKS = 2

# Scoring Weights
WEIGHT_SIMILARITY = 0.30
WEIGHT_SOURCE_DIVERSITY = 0.20
WEIGHT_SUFFICIENCY = 0.20
WEIGHT_RETRY = 0.15
WEIGHT_VERIFICATION = 0.15

# Supported Personas
PERSONA_JUNIOR_ANALYST = "Junior Analyst"
PERSONA_PORTFOLIO_MANAGER = "Portfolio Manager"
PERSONA_COMPLIANCE_REVIEWER = "Compliance Reviewer"

SUPPORTED_PERSONAS = [
    PERSONA_JUNIOR_ANALYST,
    PERSONA_PORTFOLIO_MANAGER,
    PERSONA_COMPLIANCE_REVIEWER,
]

# Query Types
QUERY_FACT_RETRIEVAL = "Fact Retrieval"
QUERY_COMPARISON = "Comparison"
QUERY_MULTI_HOP = "Multi-Hop Reasoning"
QUERY_DECISION_SUPPORT = "Decision Support"

ALL_QUERY_TYPES = [
    QUERY_FACT_RETRIEVAL,
    QUERY_COMPARISON,
    QUERY_MULTI_HOP,
    QUERY_DECISION_SUPPORT,
]
