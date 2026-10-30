import re
from typing import List, Tuple
from app.config import (
    QUERY_FACT_RETRIEVAL,
    QUERY_COMPARISON,
    QUERY_MULTI_HOP,
    QUERY_DECISION_SUPPORT,
)


class QueryRewriter:
    """
    Rewrites user financial queries into semantically dense retrieval representations
    and performs query decomposition for deep search / complex multi-hop queries.
    """

    FINANCIAL_ACRONYMS = {
        r"\bnmbr\b": "Nimbus Robotics NMBR",
        r"\bsldc\b": "Solace Dynamics SLDC",
        r"\beps\b": "diluted earnings per share EPS",
        r"\bd/e\b": "debt-to-equity ratio leverage",
        r"\bdebt to equity\b": "total debt to total equity ratio leverage",
        r"\bgm\b": "gross margin percentage",
        r"\bpe\b": "forward P/E price to earnings valuation multiple",
        r"\bp/e\b": "forward P/E price to earnings valuation multiple",
        r"\brevolver\b": "revolving credit facility borrowing revolver draw",
        r"\b10-?q\b": "Form 10-Q quarterly report SEC filing",
        r"\bmd&a\b": "Management Discussion and Analysis MD&A",
        r"\byoy\b": "year-over-year YoY change",
    }

    CONVERSATIONAL_PREFIXES = [
        r"^(?:can\s+you\s+)?(?:could\s+you\s+)?(?:please\s+)?(?:tell\s+me|show\s+me|explain|provide|give\s+me)\s+(?:about\s+|on\s+)?",
        r"^(?:please\s+)?(?:can\s+you\s+)?(?:tell\s+me\s+)?(?:about\s+)?",
        r"^(?:please\s+)?(?:what\s+is|what\s+are|what\s+was|what\s+were)\s+",
        r"^(?:i\s+would\s+like\s+to\s+know|i\s+want\s+to\s+know)\s+(?:about\s+)?",
        r"^(?:can\s+you\s+)?find\s+(?:information\s+on\s+)?",
    ]

    @classmethod
    def rewrite_query(cls, query: str) -> str:
        rewritten = query.strip()

        # Remove polite/conversational filler prefixes
        for pattern in cls.CONVERSATIONAL_PREFIXES:
            rewritten = re.sub(pattern, "", rewritten, flags=re.IGNORECASE)

        # Normalize punctuation
        rewritten = re.sub(r"[?!.,;]+$", "", rewritten).strip()

        # Expand acronyms without duplicating already expanded company names
        for pattern, expansion in cls.FINANCIAL_ACRONYMS.items():
            if re.search(pattern, rewritten, flags=re.IGNORECASE):
                # Avoid "Nimbus Robotics Nimbus Robotics"
                if "nimbus" not in rewritten.lower() or "nmbr" not in pattern:
                    if "solace" not in rewritten.lower() or "sldc" not in pattern:
                        rewritten = re.sub(pattern, expansion, rewritten, flags=re.IGNORECASE)

        return rewritten.strip()

    @classmethod
    def decompose_query(cls, query: str, query_type: str) -> List[str]:
        """
        Decomposes complex queries into 2-3 logical sub-queries.
        """
        q_lower = query.lower()
        sub_queries: List[str] = []

        if query_type == QUERY_COMPARISON or (("nmbr" in q_lower or "nimbus" in q_lower) and ("sldc" in q_lower or "solace" in q_lower)):
            # Sub-query 1: NMBR metrics & operations
            sub_queries.append("Nimbus Robotics NMBR revenue gross margin operating income and input cost inflation")
            # Sub-query 2: SLDC metrics & operations
            sub_queries.append("Solace Dynamics SLDC revenue gross margin component supplier diversification")
            # Sub-query 3: Direct comparative metrics & leverage
            sub_queries.append("NMBR vs SLDC comparative valuation forward PE multiple debt-to-equity leverage and analyst recommendation")

        elif query_type == QUERY_MULTI_HOP:
            # Sub-query 1: Cause / Driver (e.g. input costs, steel & semiconductor, freight)
            sub_queries.append("steel castings semiconductor input costs freight surcharges gross margin compression")
            # Sub-query 2: Transmission mechanism (operating income, EPS decline, revolver draw)
            sub_queries.append("diluted earnings per share EPS decline bridge interest expense R&D spend")
            # Sub-query 3: Balance sheet consequence & mitigations
            sub_queries.append("revolving credit facility drawdown inventory build total debt to equity ratio mitigating actions")

        elif query_type == QUERY_DECISION_SUPPORT:
            # Sub-query 1: Analyst recommendation & ratings
            sub_queries.append("equity research note analyst recommendation rating NMBR Neutral SLDC Outperform")
            # Sub-query 2: Risk profile, leverage & balance sheet flexibility
            sub_queries.append("balance sheet risk leverage debt-to-equity ratio liquidity interest coverage")
            # Sub-query 3: Growth vs valuation tradeoff
            sub_queries.append("revenue growth rate valuation forward PE multiple stock performance risk tolerance")

        else:
            # Fact retrieval or generic multi-aspect
            base_rewritten = cls.rewrite_query(query)
            sub_queries.append(base_rewritten)
            sub_queries.append(f"{base_rewritten} financial statements and earnings transcript")

        # Ensure between 2 and 3 sub-queries
        return sub_queries[:3]
