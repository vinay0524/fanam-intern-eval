def classify_query(query: str) -> str:
    query_lower = query.lower()

    if any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
        return "comparison"

    if any(word in query_lower for word in ["why", "explain", "reason"]):
        return "analytical"

    if any(word in query_lower for word in ["calculate", "percentage", "growth"]):
        return "calculation"

    if any(word in query_lower for word in ["revenue", "profit", "financial", "report"]):
        return "financial"

    return "general"