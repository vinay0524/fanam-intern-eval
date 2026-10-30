import re
from typing import List, Set
from app.config import (
    SUFFICIENCY_SIMILARITY_THRESHOLD,
    MIN_SUFFICIENCY_CHUNKS,
)
from app.models import DocumentChunk, SufficiencyResult


class RetrievalSufficiencyAssessor:
    """
    Evaluates whether retrieved chunks provide adequate coverage, volume, and similarity
    to faithfully answer the financial prompt.
    """

    FINANCIAL_STOPWORDS = {
        "what", "when", "where", "which", "who", "whom", "whose", "why",
        "how", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "the", "a", "an",
        "and", "or", "but", "if", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below",
        "to", "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there",
        "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "can", "will", "just", "should", "now"
    }

    @classmethod
    def extract_salient_terms(cls, query: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\.%]+\b", query.lower())
        salient = {w for w in words if w not in cls.FINANCIAL_STOPWORDS and len(w) > 1}
        return salient

    @classmethod
    def evaluate(cls, query: str, chunks: List[DocumentChunk]) -> SufficiencyResult:
        if not chunks:
            return SufficiencyResult(
                is_sufficient=False,
                score=0.0,
                top_similarity=0.0,
                chunk_count=0,
                coverage_score=0.0,
                reason="Zero chunks retrieved",
            )

        chunk_count = len(chunks)
        top_similarity = float(chunks[0].metadata.__dict__.get("dense_similarity", 0.0))
        
        # Combined text of all retrieved chunks
        corpus_text = " ".join([chunk.content.lower() for chunk in chunks])

        # Key salient terms from query
        salient_terms = cls.extract_salient_terms(query)
        if salient_terms:
            covered_terms = {term for term in salient_terms if term in corpus_text}
            coverage_score = len(covered_terms) / len(salient_terms)
        else:
            coverage_score = 1.0

        # Criteria checks
        has_volume = chunk_count >= MIN_SUFFICIENCY_CHUNKS
        has_similarity = top_similarity >= SUFFICIENCY_SIMILARITY_THRESHOLD
        has_coverage = coverage_score >= 0.50

        is_sufficient = has_volume and has_similarity and has_coverage

        # Compute continuous sufficiency score (0.0 - 1.0)
        volume_component = min(1.0, chunk_count / 3.0)
        sim_component = min(1.0, max(0.0, top_similarity))
        sufficiency_score = (0.40 * coverage_score) + (0.35 * sim_component) + (0.25 * volume_component)

        reason_parts = []
        if not has_volume:
            reason_parts.append(f"insufficient chunk volume ({chunk_count} < {MIN_SUFFICIENCY_CHUNKS})")
        if not has_similarity:
            reason_parts.append(f"low semantic similarity ({top_similarity:.2f} < {SUFFICIENCY_SIMILARITY_THRESHOLD})")
        if not has_coverage:
            missing = salient_terms - covered_terms if salient_terms else set()
            reason_parts.append(f"low query entity coverage ({coverage_score:.0%}, missing: {list(missing)[:3]})")

        reason = "Retrieval sufficient." if is_sufficient else f"Retrieval insufficient: {'; '.join(reason_parts)}"

        return SufficiencyResult(
            is_sufficient=is_sufficient,
            score=round(sufficiency_score, 4),
            top_similarity=round(top_similarity, 4),
            chunk_count=chunk_count,
            coverage_score=round(coverage_score, 4),
            reason=reason,
        )

