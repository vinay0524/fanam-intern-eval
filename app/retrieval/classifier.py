import re
from typing import Dict, Any
from app.config import (
    QUERY_FACT_RETRIEVAL,
    QUERY_COMPARISON,
    QUERY_MULTI_HOP,
    QUERY_DECISION_SUPPORT,
)


class QueryClassifier:
    """
    Classifies queries into one of four financial research archetypes:
    - Fact Retrieval: Direct lookup of financial figures, dates, officers, single metrics.
    - Comparison: Multi-entity or multi-period comparative evaluations.
    - Multi-Hop Reasoning: Causal, explanatory, or chained financial relationships.
    - Decision Support: Investment recommendations, strategic allocation, rating decisions.
    """

    COMPARISON_INDICATORS = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference between\b",
        r"\bbetter\b",
        r"\bworse\b",
        r"\bhigher than\b",
        r"\blower than\b",
        r"\brelative to\b",
        r"\bboth companies\b",
        r"\bnmbr (?:and|vs) sldc\b",
        r"\bsldc (?:and|vs) nmbr\b",
    ]

    MULTI_HOP_INDICATORS = [
        r"\bwhy did\b",
        r"\bhow did\b",
        r"\bhow does\b",
        r"\bimpact of\b",
        r"\bimpact on\b",
        r"\baffect(?:ed|ing)?\b",
        r"\bcaused by\b",
        r"\bcascading\b",
        r"\bchain of\b",
        r"\bcorrelation\b",
        r"\bdriver of\b",
        r"\bprimary driver\b",
        r"\battributable to\b",
        r"\bdue to\b",
        r"\bresulted in\b",
        r"\bconsequence\b",
        r"\btransmission mechanism\b",
        r"\bhow (?:input cost|inflation|debt|margin) (?:led to|impacted)\b",
    ]

    DECISION_SUPPORT_INDICATORS = [
        r"\bshould\b",
        r"\brecommend\b",
        r"\brecommendation\b",
        r"\brating\b",
        r"\boutperform\b",
        r"\bneutral\b",
        r"\bunderperform\b",
        r"\bbuy\b",
        r"\bsell\b",
        r"\bhold\b",
        r"\binvest(?:or|ment)?\b",
        r"\ballocat(?:e|ion)\b",
        r"\brisk tolerance\b",
        r"\brisk-averse\b",
        r"\bportfolio manager\b",
        r"\bwhich (?:stock|company|asset) is (?:better|favored|preferable)\b",
        r"\bstrategy\b",
        r"\bthesis\b",
    ]

    @classmethod
    def classify(cls, query: str) -> str:
        q_clean = query.strip().lower()

        # Decision Support check (ratings, recommendations, investor suitability)
        for pattern in cls.DECISION_SUPPORT_INDICATORS:
            if re.search(pattern, q_clean):
                return QUERY_DECISION_SUPPORT

        # Comparison check (explicit comparison, mentions both NMBR and SLDC)
        both_tickers = ("nmbr" in q_clean or "nimbus" in q_clean) and ("sldc" in q_clean or "solace" in q_clean)
        for pattern in cls.COMPARISON_INDICATORS:
            if re.search(pattern, q_clean) or both_tickers:
                return QUERY_COMPARISON

        # Multi-Hop Reasoning check (causality, transmission, explanatory relationships)
        for pattern in cls.MULTI_HOP_INDICATORS:
            if re.search(pattern, q_clean):
                return QUERY_MULTI_HOP

        # Default is Fact Retrieval
        return QUERY_FACT_RETRIEVAL

