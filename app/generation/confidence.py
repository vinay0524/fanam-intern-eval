from typing import List
from app.config import (
    WEIGHT_SIMILARITY,
    WEIGHT_SOURCE_DIVERSITY,
    WEIGHT_SUFFICIENCY,
    WEIGHT_RETRY,
    WEIGHT_VERIFICATION,
)
from app.models import (
    DocumentChunk,
    SufficiencyResult,
    VerificationResult,
    ConfidenceBreakdown,
)


class ConfidenceScorer:
    """
    Computes the strict weighted confidence score:
    Confidence = (0.30 * Similarity) + (0.20 * Source_Diversity) + (0.20 * Sufficiency)
                 + (0.15 * Retry_Score) + (0.15 * Verification_Success)
    """

    @classmethod
    def calculate_source_diversity(cls, chunks: List[DocumentChunk]) -> float:
        if not chunks:
            return 0.0
        unique_docs = {chunk.metadata.document_name for chunk in chunks}
        # If at least 2 distinct documents are cited, diversity is high (1.0). 1 document yields 0.50
        diversity = min(1.0, len(unique_docs) / 2.0)
        return round(diversity, 4)

    @classmethod
    def calculate_retry_score(cls, attempts: int, is_sufficient: bool) -> float:
        if attempts == 1 and is_sufficient:
            return 1.0
        elif attempts == 2 and is_sufficient:
            return 0.70
        elif attempts == 3 and is_sufficient:
            return 0.40
        else:
            return 0.10

    @classmethod
    def compute_confidence(
        cls,
        chunks: List[DocumentChunk],
        sufficiency: SufficiencyResult,
        verification: VerificationResult,
        attempts: int,
    ) -> ConfidenceBreakdown:
        # 1. Similarity (Top chunk dense cosine similarity)
        top_sim = float(chunks[0].metadata.__dict__.get("dense_similarity", 0.0)) if chunks else 0.0
        # Normalize and scale reasonable cosine similarities (e.g. 0.4 - 0.8 to a normalized 0.0 - 1.0)
        normalized_sim = min(1.0, max(0.0, top_sim))

        # 2. Source Diversity
        diversity = cls.calculate_source_diversity(chunks)

        # 3. Sufficiency Score
        suff_score = sufficiency.score

        # 4. Retry Score
        retry_score = cls.calculate_retry_score(attempts, sufficiency.is_sufficient)

        # 5. Verification Success Rate
        verif_success = verification.success_rate

        # Total Weighted Calculation
        total = (
            (WEIGHT_SIMILARITY * normalized_sim)
            + (WEIGHT_SOURCE_DIVERSITY * diversity)
            + (WEIGHT_SUFFICIENCY * suff_score)
            + (WEIGHT_RETRY * retry_score)
            + (WEIGHT_VERIFICATION * verif_success)
        )

        total_clamped = round(min(1.0, max(0.0, total)), 2)

        return ConfidenceBreakdown(
            similarity=round(normalized_sim, 4),
            source_diversity=round(diversity, 4),
            sufficiency=round(suff_score, 4),
            retry_score=round(retry_score, 4),
            verification_success=round(verif_success, 4),
            total_confidence=total_clamped,
        )

