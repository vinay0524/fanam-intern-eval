from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., description="User query or financial research prompt")
    persona: str = Field(
        default="Junior Analyst",
        description="Target persona: 'Junior Analyst', 'Portfolio Manager', or 'Compliance Reviewer'",
    )
    deep_search: bool = Field(
        default=False,
        description="Flag to force query decomposition and multi-hop hybrid search",
    )


class ChunkMetadata(BaseModel):
    document_name: str
    document_type: str
    page_number: int
    chunk_id: str


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    score: Optional[float] = None


class AgentResponse(BaseModel):
    session_id: str
    query_type: str
    original_query: str
    rewritten_query: str
    deep_search_used: bool
    retrieval_attempts: int
    retrieval_sufficient: bool
    confidence_score: float
    used_chunks: List[str]
    answer: str


class SufficiencyResult(BaseModel):
    is_sufficient: bool
    score: float
    top_similarity: float
    chunk_count: int
    coverage_score: float
    reason: str


class VerificationResult(BaseModel):
    success_rate: float
    verified_claims: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    cleaned_answer: Optional[str] = None


class ConfidenceBreakdown(BaseModel):
    similarity: float
    source_diversity: float
    sufficiency: float
    retry_score: float
    verification_success: float
    total_confidence: float

