from pydantic import BaseModel, Field


class FinanceQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str = "default"
    persona: str = "financial_analyst"
    deep_search: bool = False


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class IngestRequest(BaseModel):
    rebuild_index: bool = True


class ClassifyRequest(BaseModel):
    query: str = Field(min_length=1)


class RewriteRequest(BaseModel):
    query: str = Field(min_length=1)
    history: str | None = None


class DecomposeRequest(BaseModel):
    query: str = Field(min_length=1)


class FinanceQueryResponse(BaseModel):
    session_id: str
    query: str
    query_type: str
    answer: str
    sources: list
    confidence: float
    verification: dict
    retrieval_count: int
    retry_attempts: int
