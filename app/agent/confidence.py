STRONG_EVIDENCE_THRESHOLD = 0.85
MODERATE_EVIDENCE_THRESHOLD = 0.65


def get_confidence_level(score):
    if score >= STRONG_EVIDENCE_THRESHOLD:
        return "strong"
    elif score >= MODERATE_EVIDENCE_THRESHOLD:
        return "moderate"
    return "weak"


STRONG_EVIDENCE_THRESHOLD = 0.85
MODERATE_EVIDENCE_THRESHOLD = 0.65


def get_confidence_level(score):
    if score >= STRONG_EVIDENCE_THRESHOLD:
        return "strong"
    elif score >= MODERATE_EVIDENCE_THRESHOLD:
        return "moderate"
    return "weak"


def calculate_confidence(results: list, verification: dict) -> float:
    if not results:
        return 0.0

    confidence = 0.5

    if len(results) >= 3:
        confidence += 0.2

    if verification.get("verified", False):
        confidence += 0.2

    support_ratio = verification.get("support_ratio", 0)
    confidence += min(support_ratio * 0.1, 0.1)

    return round(min(confidence, 1.0), 2)
