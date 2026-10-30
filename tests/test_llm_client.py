import os
import pytest
from unittest.mock import patch, MagicMock

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
except ImportError:
    pass

try:
    import faiss
except ImportError:
    pass

from app.config import (
    PERSONA_JUNIOR_ANALYST,
    PERSONA_PORTFOLIO_MANAGER,
    PERSONA_COMPLIANCE_REVIEWER,
)
from app.models import DocumentChunk, ChunkMetadata
from app.generation.llm_client import LLMClient
from app.generation.generator import PersonaResponseGenerator


@pytest.fixture
def sample_test_chunks():
    m1 = ChunkMetadata(
        document_name="Nimbus_Robotics_10Q_Q2_2026.pdf",
        document_type="pdf",
        page_number=1,
        chunk_id="chunk_1",
    )
    c1 = DocumentChunk(
        chunk_id="chunk_1",
        content="Net revenue for Q2 2026 was $412.6 million, an increase of 9.4% YoY. Gross margin declined to 38.2% from 43.7%. Diluted EPS was $0.61.",
        metadata=m1,
        score=0.10,
    )
    return [c1]


def test_llm_client_config():
    # If no keys in environment
    with patch("app.generation.llm_client.GEMINI_API_KEY", ""), patch(
        "app.generation.llm_client.OPENAI_API_KEY", ""
    ):
        assert LLMClient.is_configured() is False
        assert LLMClient.get_active_provider() is None

    # When Gemini key is provided
    with patch("app.generation.llm_client.GEMINI_API_KEY", "test_gemini_key"):
        assert LLMClient.is_configured() is True
        assert "Gemini" in LLMClient.get_active_provider()

    # When OpenAI key is provided
    with patch("app.generation.llm_client.GEMINI_API_KEY", ""), patch(
        "app.generation.llm_client.OPENAI_API_KEY", "test_openai_key"
    ):
        assert LLMClient.is_configured() is True
        assert "OpenAI" in LLMClient.get_active_provider()


def test_llm_prompt_building(sample_test_chunks):
    prompts = LLMClient.build_prompt(
        persona=PERSONA_JUNIOR_ANALYST,
        query="Explain NMBR revenue growth",
        query_type="Fact Retrieval",
        chunks=sample_test_chunks,
    )
    assert "system_prompt" in prompts
    assert "user_prompt" in prompts
    assert "Junior Analyst" in prompts["system_prompt"]
    assert "Chunk 1" in prompts["user_prompt"]
    assert "412.6" in prompts["user_prompt"]


def test_llm_fallback_when_keys_missing(sample_test_chunks):
    with patch("app.generation.llm_client.GEMINI_API_KEY", ""), patch(
        "app.generation.llm_client.OPENAI_API_KEY", ""
    ):
        res = LLMClient.generate_persona_response(
            persona=PERSONA_JUNIOR_ANALYST,
            query="Explain NMBR revenue growth",
            query_type="Fact Retrieval",
            chunks=sample_test_chunks,
        )
        assert res is None

        # Verify generator falls back to deterministic template synthesis
        answer, used_chunks = PersonaResponseGenerator.generate(
            query="Explain NMBR revenue growth",
            query_type="Fact Retrieval",
            persona=PERSONA_JUNIOR_ANALYST,
            chunks=sample_test_chunks,
        )
        assert len(answer) > 0
        assert "Junior Analyst" in answer
        assert len(used_chunks) > 0


def test_llm_fallback_on_api_error(sample_test_chunks):
    # Simulate Gemini exception (e.g., quota exceeded or network down)
    with patch("app.generation.llm_client.GEMINI_API_KEY", "mock_key_with_error"):
        with patch("google.genai.Client", side_effect=RuntimeError("Network connection reset")):
            res = LLMClient.generate_persona_response(
                persona=PERSONA_PORTFOLIO_MANAGER,
                query="Investment thesis on NMBR",
                query_type="Decision Support",
                chunks=sample_test_chunks,
            )
            assert res is None

            # Verify generator still produces grounded answer via fallback
            answer, used_chunks = PersonaResponseGenerator.generate(
                query="Investment thesis on NMBR",
                query_type="Decision Support",
                persona=PERSONA_PORTFOLIO_MANAGER,
                chunks=sample_test_chunks,
            )
            assert "Portfolio Manager" in answer
            assert len(used_chunks) > 0


def test_llm_successful_mock_generation(sample_test_chunks):
    mock_llm_text = (
        "### Financial Research Analysis: Revenue Performance\n\n"
        "**Persona Perspective**: Junior Analyst — Step-by-Step Financial Reasoning & Accounting Deconstruction.\n\n"
        "#### 1. Core Financial Findings & Direct Evidence\n"
        "- Net revenue was $412.6 million [Chunk 1].\n\n"
        "#### 2. Step-by-Step Financial Concepts & Accounting Mechanics\n"
        "- Gross margin was 38.2% and EPS was $0.61 [Chunk 1].\n\n"
        "#### 3. Management Remediation Actions\n"
        "- Expanding operations.\n\n"
        "#### 4. Source Chunk Citations\n"
        "- [Chunk 1]: Nimbus_Robotics_10Q_Q2_2026.pdf"
    )

    with patch(
        "app.generation.llm_client.LLMClient.generate_persona_response",
        return_value=mock_llm_text,
    ):
        answer, used_chunks = PersonaResponseGenerator.generate(
            query="Revenue performance",
            query_type="Fact Retrieval",
            persona=PERSONA_JUNIOR_ANALYST,
            chunks=sample_test_chunks,
        )
        assert answer == mock_llm_text
        assert "chunk_1" in used_chunks

