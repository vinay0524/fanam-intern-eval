def create_plan(query: str, query_type: str) -> dict:
    if query_type == "comparison":
        return {
            "strategy": "multi_step",
            "decompose": True,
            "rewrite": False,
            "top_k": 8
        }

    if query_type == "analytical":
        return {
            "strategy": "deep_search",
            "decompose": True,
            "rewrite": False,
            "top_k": 8
        }

    return {
        "strategy": "standard",
        "decompose": False,
        "rewrite": False,
        "top_k": 5
    }