import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except ImportError:
    pass

try:
    import faiss
except ImportError:
    pass

import pytest
from app.server import health_check, finance_agent_endpoint, trigger_ingestion, serve_ui, agent
from app.models import AgentRequest, AgentResponse


@pytest.fixture(scope="module", autouse=True)
def initialize_system():
    if not agent.is_initialized:
        agent.initialize()


def test_serve_ui():
    res = serve_ui()
    assert res.status_code == 200
    assert b"Financial Research Assistant" in res.body
    assert b"Junior Analyst" in res.body


def test_health_endpoint():
    data = health_check()
    assert data["status"] == "healthy"
    assert data["agent_initialized"] is True
    assert data["indexed_chunks"] > 0


def test_finance_agent_junior_analyst():
    req = AgentRequest(
        session_id="api_test_001",
        query="What was Nimbus Robotics net revenue and gross margin in Q2 2026?",
        persona="Junior Analyst",
        deep_search=False,
    )
    res = finance_agent_endpoint(req)
    assert isinstance(res, AgentResponse)

    data = res.model_dump()
    required_keys = [
        "session_id",
        "query_type",
        "original_query",
        "rewritten_query",
        "deep_search_used",
        "retrieval_attempts",
        "retrieval_sufficient",
        "confidence_score",
        "used_chunks",
        "answer",
    ]
    for key in required_keys:
        assert key in data, f"Key '{key}' must be present in response payload"

    assert data["session_id"] == "api_test_001"
    assert data["query_type"] == "Fact Retrieval"
    assert data["retrieval_attempts"] >= 1
    assert data["retrieval_sufficient"] is True
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert isinstance(data["used_chunks"], list)
    assert len(data["used_chunks"]) > 0
    assert "Junior Analyst" in data["answer"]


def test_finance_agent_portfolio_manager():
    req = AgentRequest(
        session_id="api_test_002",
        query="Compare gross margin and leverage between NMBR and SLDC",
        persona="Portfolio Manager",
        deep_search=True,
    )
    res = finance_agent_endpoint(req)
    data = res.model_dump()

    assert data["session_id"] == "api_test_002"
    assert data["query_type"] == "Comparison"
    assert data["deep_search_used"] is True
    assert "Portfolio Manager" in data["answer"]


def test_finance_agent_compliance_reviewer():
    req = AgentRequest(
        session_id="api_test_003",
        query="Why did gross margin fall and what are the debt risk factors?",
        persona="Compliance Reviewer",
        deep_search=False,
    )
    res = finance_agent_endpoint(req)
    data = res.model_dump()

    assert data["session_id"] == "api_test_003"
    assert data["query_type"] in ["Multi-Hop Reasoning", "Fact Retrieval"]
    assert "Compliance Reviewer" in data["answer"]
    assert "Limitations" in data["answer"] or "Disclosures" in data["answer"]
