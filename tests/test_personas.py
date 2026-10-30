import pytest
from app.config import (
    PERSONA_JUNIOR_ANALYST,
    PERSONA_PORTFOLIO_MANAGER,
    PERSONA_COMPLIANCE_REVIEWER,
)
from app.models import DocumentChunk, ChunkMetadata, SufficiencyResult, VerificationResult
from app.generation.generator import PersonaResponseGenerator
from app.generation.verifier import FaithfulnessVerifier
from app.generation.confidence import ConfidenceScorer


@pytest.fixture
def sample_chunks():
    m1 = ChunkMetadata(
        document_name="Nimbus_Robotics_10Q_Q2_2026.pdf",
        document_type="pdf",
        page_number=1,
        chunk_id="chunk_1",
    )
    m2 = ChunkMetadata(
        document_name="NMBR_Q2_2026_Earnings_Call_Transcript.txt",
        document_type="txt",
        page_number=1,
        chunk_id="chunk_2",
    )
    m3 = ChunkMetadata(
        document_name="Analyst_Note_NMBR_vs_SLDC.docx",
        document_type="docx",
        page_number=1,
        chunk_id="chunk_3",
    )

    c1 = DocumentChunk(
        chunk_id="chunk_1",
        content="Net revenue for Q2 2026 was $412.6 million, an increase of 9.4% YoY. Gross margin declined to 38.2% from 43.7%. Diluted EPS was $0.61 versus $0.89.",
        metadata=m1,
        score=0.10,
    )
    c1.metadata.__dict__["dense_similarity"] = 0.82

    c2 = DocumentChunk(
        chunk_id="chunk_2",
        content="Total debt was $612.4 million ($162.4 million drawn on revolver) against total equity of $498.1 million. Debt-to-equity ratio reached 1.23x. Input costs rose 21%.",
        metadata=m2,
        score=0.09,
    )
    c2.metadata.__dict__["dense_similarity"] = 0.78

    c3 = DocumentChunk(
        chunk_id="chunk_3",
        content="Solace Dynamics SLDC rated Outperform, NMBR rated Neutral. Forward P/E is 18.4x for NMBR ($56.75) and 15.1x for SLDC ($79.10). SLDC debt-to-equity is 0.58x, gross margin 39.1%.",
        metadata=m3,
        score=0.08,
    )
    c3.metadata.__dict__["dense_similarity"] = 0.75

    return [c1, c2, c3]


def test_junior_analyst_generation(sample_chunks):
    answer, used_chunks = PersonaResponseGenerator.generate(
        query="Explain NMBR Q2 2026 financial performance and margin drivers",
        query_type="Multi-Hop Reasoning",
        persona=PERSONA_JUNIOR_ANALYST,
        chunks=sample_chunks,
    )
    assert "Junior Analyst" in answer
    assert "Step-by-Step" in answer or "Step 1" in answer
    assert "Gross Margin" in answer or "COGS" in answer
    assert "[Chunk" in answer
    assert len(used_chunks) > 0


def test_portfolio_manager_generation(sample_chunks):
    answer, used_chunks = PersonaResponseGenerator.generate(
        query="Investment thesis comparing NMBR and SLDC",
        query_type="Comparison",
        persona=PERSONA_PORTFOLIO_MANAGER,
        chunks=sample_chunks,
    )
    assert "Portfolio Manager" in answer
    assert "Valuation" in answer or "P/E" in answer
    assert "Balance Sheet" in answer or "Leverage" in answer
    assert len(used_chunks) > 0


def test_compliance_reviewer_generation(sample_chunks):
    answer, used_chunks = PersonaResponseGenerator.generate(
        query="Audit and compliance check of NMBR Q2 2026 figures",
        query_type="Fact Retrieval",
        persona=PERSONA_COMPLIANCE_REVIEWER,
        chunks=sample_chunks,
    )
    assert "Compliance Reviewer" in answer
    assert "Audit Trail" in answer
    assert "Risk Disclosures" in answer or "Limitations" in answer
    assert len(used_chunks) > 0


def test_faithfulness_verifier(sample_chunks):
    grounded_text = "Net revenue was $412.6 million in Q2 2026. Gross margin was 38.2%."
    res = FaithfulnessVerifier.verify(grounded_text, sample_chunks)
    assert res.success_rate == 1.0
    assert len(res.unsupported_claims) == 0

    ungrounded_text = "Net revenue was $9999.9 million in Q2 2026. Gross margin was 99.9%."
    res_fake = FaithfulnessVerifier.verify(ungrounded_text, sample_chunks)
    assert len(res_fake.unsupported_claims) > 0
    assert res_fake.success_rate < 1.0


def test_weighted_confidence_formula(sample_chunks):
    suff = SufficiencyResult(
        is_sufficient=True,
        score=0.90,
        top_similarity=0.82,
        chunk_count=3,
        coverage_score=0.95,
        reason="Sufficient",
    )
    verif = VerificationResult(
        success_rate=1.0,
        verified_claims=["Claim 1", "Claim 2"],
        unsupported_claims=[],
    )
    breakdown = ConfidenceScorer.compute_confidence(
        chunks=sample_chunks,
        sufficiency=suff,
        verification=verif,
        attempts=1,
    )

    # Expected: (0.30 * 0.82) + (0.20 * 1.0) + (0.20 * 0.90) + (0.15 * 1.0) + (0.15 * 1.0)
    # = 0.246 + 0.20 + 0.18 + 0.15 + 0.15 = 0.926 -> 0.93
    assert 0.90 <= breakdown.total_confidence <= 0.95
    assert breakdown.similarity == 0.82
    assert breakdown.source_diversity == 1.0
    assert breakdown.retry_score == 1.0
    assert breakdown.verification_success == 1.0

