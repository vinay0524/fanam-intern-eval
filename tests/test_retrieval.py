import pytest
from app.config import (
    QUERY_FACT_RETRIEVAL,
    QUERY_COMPARISON,
    QUERY_MULTI_HOP,
    QUERY_DECISION_SUPPORT,
)
from app.retrieval.classifier import QueryClassifier
from app.retrieval.rewriter import QueryRewriter
from app.retrieval.sufficiency import RetrievalSufficiencyAssessor
from app.models import DocumentChunk, ChunkMetadata


def test_query_classifier():
    fact_q = "What was Nimbus Robotics net revenue in Q2 2026?"
    assert QueryClassifier.classify(fact_q) == QUERY_FACT_RETRIEVAL

    comp_q = "Compare the gross margins and leverage of NMBR vs SLDC."
    assert QueryClassifier.classify(comp_q) == QUERY_COMPARISON

    multihop_q = "Why did input cost inflation impact EPS and subsequently drive debt-to-equity leverage?"
    assert QueryClassifier.classify(multihop_q) == QUERY_MULTI_HOP

    decision_q = "Which stock should a risk-averse portfolio manager recommend for capital preservation?"
    assert QueryClassifier.classify(decision_q) == QUERY_DECISION_SUPPORT


def test_query_rewriter():
    raw_query = "Can you please tell me about NMBR EPS and D/E ratio?"
    rewritten = QueryRewriter.rewrite_query(raw_query)

    assert "can you please tell me" not in rewritten.lower()
    assert "nimbus robotics nmbr" in rewritten.lower() or "nmbr" in rewritten.lower()
    assert "diluted earnings per share" in rewritten.lower()


def test_query_decomposition():
    comp_q = "Compare gross margins of NMBR vs SLDC"
    sub_qs = QueryRewriter.decompose_query(comp_q, QUERY_COMPARISON)
    assert 2 <= len(sub_qs) <= 3
    assert any("nmbr" in sq.lower() for sq in sub_qs)
    assert any("sldc" in sq.lower() for sq in sub_qs)

    multihop_q = "How did input cost increases lead to revolver draws and higher leverage?"
    sub_multihop = QueryRewriter.decompose_query(multihop_q, QUERY_MULTI_HOP)
    assert 2 <= len(sub_multihop) <= 3


def test_retrieval_sufficiency():
    dummy_metadata = ChunkMetadata(
        document_name="Nimbus_Robotics_10Q_Q2_2026.pdf",
        document_type="pdf",
        page_number=1,
        chunk_id="chunk_1",
    )
    c1 = DocumentChunk(
        chunk_id="chunk_1",
        content="Net revenue for Q2 2026 was $412.6 million. Gross margin was 38.2%. Total debt was $612.4 million.",
        metadata=dummy_metadata,
        score=0.08,
    )
    c1.metadata.__dict__["dense_similarity"] = 0.75

    c2 = DocumentChunk(
        chunk_id="chunk_2",
        content="Diluted earnings per share was $0.61 down from $0.89. Debt-to-equity ratio was 1.23x.",
        metadata=dummy_metadata,
        score=0.07,
    )
    c2.metadata.__dict__["dense_similarity"] = 0.70

    # Sufficient case
    res = RetrievalSufficiencyAssessor.evaluate("revenue gross margin debt-to-equity", [c1, c2])
    assert res.is_sufficient is True
    assert res.score > 0.6
    assert res.chunk_count == 2

    # Insufficient case: empty chunks
    res_empty = RetrievalSufficiencyAssessor.evaluate("revenue", [])
    assert res_empty.is_sufficient is False
    assert res_empty.chunk_count == 0

