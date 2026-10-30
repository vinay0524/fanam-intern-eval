import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"

class Settings:
    @property
    def GEMINI_API_KEY(self) -> str:
        load_dotenv(dotenv_path=env_path, override=True)
        return os.getenv("GEMINI_API_KEY", "").strip()

    @property
    def GEMINI_EMBEDDING_MODEL(self) -> str:
        load_dotenv(dotenv_path=env_path, override=True)
        return os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()

    @property
    def GEMINI_LLM_MODEL(self) -> str:
        load_dotenv(dotenv_path=env_path, override=True)
        return os.getenv("GEMINI_LLM_MODEL", "models/gemini-3.5-flash-lite").strip()

    @property
    def PINECONE_API_KEY(self) -> str:
        load_dotenv(dotenv_path=env_path, override=True)
        return os.getenv("PINECONE_API_KEY", "").strip()

    @property
    def PINECONE_INDEX_NAME(self) -> str:
        load_dotenv(dotenv_path=env_path, override=True)
        return os.getenv("PINECONE_INDEX_NAME", "financial-rag").strip()

    @property
    def PINECONE_CLOUD(self) -> str:
        return os.getenv("PINECONE_CLOUD", "aws").strip()

    @property
    def PINECONE_REGION(self) -> str:
        return os.getenv("PINECONE_REGION", "us-east-1").strip()

    @property
    def DEFAULT_ALPHA(self) -> float:
        return float(os.getenv("DEFAULT_ALPHA", "0.5"))

    @property
    def PORT(self) -> int:
        return int(os.getenv("PORT", "8000"))

    @property
    def HOST(self) -> str:
        return os.getenv("HOST", "0.0.0.0")

    def validate_keys(self) -> dict:
        """Returns the readiness status of external API keys dynamically."""
        missing = []
        gemini_key = self.GEMINI_API_KEY
        pinecone_key = self.PINECONE_API_KEY

        if not gemini_key or "your_" in gemini_key:
            missing.append("GEMINI_API_KEY")
        if not pinecone_key or "your_" in pinecone_key:
            missing.append("PINECONE_API_KEY")
        
        return {
            "configured": len(missing) == 0,
            "missing_keys": missing
        }

settings = Settings()
