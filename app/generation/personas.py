from typing import Dict, Any, List
from app.config import (
    PERSONA_JUNIOR_ANALYST,
    PERSONA_PORTFOLIO_MANAGER,
    PERSONA_COMPLIANCE_REVIEWER,
)
from app.models import DocumentChunk


class PersonaGuidelines:
    @staticmethod
    def get_persona_instructions(persona: str) -> Dict[str, Any]:
        p = persona.strip()
        if p.lower() in ["portfolio manager", "portfoliomanager", "pm"]:
            return {
                "name": PERSONA_PORTFOLIO_MANAGER,
                "tone": "Executive, concise, valuation and risk-focused",
                "rules": [
                    "Lead with the investment bottom line and key takeaways.",
                    "Focus heavily on forward valuation multiples (P/E), revenue growth tradeoffs, and balance-sheet solvency.",
                    "Highlight leverage (D/E ratio) and interest coverage as risk constraints.",
                    "Avoid unnecessary fluff; provide sharp, capital-allocation-oriented insights.",
                ],
            }
        elif p.lower() in ["compliance reviewer", "compliancereviewer", "compliance"]:
            return {
                "name": PERSONA_COMPLIANCE_REVIEWER,
                "tone": "Rigorous, auditable, cautious, and compliance-driven",
                "rules": [
                    "Maintain strict audit trails by citing every chunk and originating document name.",
                    "State explicit limitations of the available information (e.g. unaudited quarterly 10-Q excerpts, forward-looking statements).",
                    "Include explicit data confidence caveats and risk factor disclosures (e.g. interest rate sensitivity, single-source supplier risks).",
                    "Do not extrapolate or speculate beyond verified source chunks.",
                ],
            }
        else:
            return {
                "name": PERSONA_JUNIOR_ANALYST,
                "tone": "Detailed, didactic, step-by-step, analytical",
                "rules": [
                    "Provide comprehensive step-by-step reasoning explaining the underlying mechanics.",
                    "Explain financial concepts clearly (e.g. gross margin drivers, EPS impact calculation, revolver leverage mechanics).",
                    "Include source chunk citations strictly formatted as [Chunk X] throughout the text.",
                    "Ensure transparent bridge calculations and metric reconciliations.",
                ],
            }

