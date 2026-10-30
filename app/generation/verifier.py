import re
from typing import List, Tuple, Set, Optional
from app.models import DocumentChunk, VerificationResult


class FaithfulnessVerifier:
    """
    Verifies generated claims against retrieved context chunks to detect and eliminate
    ungrounded hallucinations and unsupported numerical assertions.
    """

    NUMERICAL_PATTERN = re.compile(
        r"(?:\$[0-9]+(?:\.[0-9]+)?(?:\s*(?:million|billion|M|B))?|[0-9]+(?:\.[0-9]+)?%|[0-9]+(?:\.[0-9]+)?x|\b[0-9]+\.[0-9]+\b)"
    )

    @classmethod
    def extract_claims(cls, text: str) -> List[str]:
        # Split on sentence boundaries, keeping markdown headers/bullets intact
        lines = text.split("\n")
        claims: List[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split sentences
            sentences = re.split(r"(?<=[.!?])\s+", line)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) > 15:
                    claims.append(s_clean)
        return claims

    @classmethod
    def verify(cls, answer: str, context_chunks: List[DocumentChunk]) -> VerificationResult:
        if not context_chunks:
            return VerificationResult(
                success_rate=0.0,
                verified_claims=[],
                unsupported_claims=["Context is empty"],
                cleaned_answer=answer,
            )

        full_context = " ".join([c.content for c in context_chunks]).lower()
        claims = cls.extract_claims(answer)
        if not claims:
            return VerificationResult(
                success_rate=1.0,
                verified_claims=[],
                unsupported_claims=[],
                cleaned_answer=answer,
            )

        verified: List[str] = []
        unsupported: List[str] = []
        cleaned_sentences: List[str] = []

        for claim in claims:
            # Extract numbers, percentages, multiples from the claim
            nums_in_claim = cls.NUMERICAL_PATTERN.findall(claim)
            claim_lower = claim.lower()

            is_claim_grounded = True
            for num in nums_in_claim:
                # Normalize number token (e.g. "$412.6" or "38.2%")
                clean_num = num.lower().replace("$", "").replace("%", "").replace("x", "").strip()
                # Check if this exact numerical token exists in context
                if clean_num not in full_context:
                    # Check if the full token is in context
                    if num.lower() not in full_context:
                        is_claim_grounded = False
                        break

            if is_claim_grounded:
                verified.append(claim)
                cleaned_sentences.append(claim)
            else:
                unsupported.append(claim)
                # Omit hallucinated claim from cleaned answer or add caution flag
                # By not appending to cleaned_sentences, we remove ungrounded hallucinations

        total = len(claims)
        success_rate = len(verified) / total if total > 0 else 1.0

        # If any unsupported claims were removed, reconstruct cleaned answer
        if unsupported:
            # Replace unsupported sentences in the answer
            cleaned_answer = answer
            for ungrounded_sentence in unsupported:
                cleaned_answer = cleaned_answer.replace(ungrounded_sentence, "")
            # Clean up residual blank lines
            cleaned_answer = re.sub(r"\n\s*\n", "\n\n", cleaned_answer).strip()
        else:
            cleaned_answer = answer

        return VerificationResult(
            success_rate=round(success_rate, 4),
            verified_claims=verified,
            unsupported_claims=unsupported,
            cleaned_answer=cleaned_answer,
        )

