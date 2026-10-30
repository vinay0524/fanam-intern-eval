
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(MODEL_NAME)

    return _reranker


def rerank_results(
    query: str,
    results: list,
    top_k: int = 5
) -> list:

    if not results:
        return []

    model = get_reranker()

    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = model.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):
        item = result.copy()
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked[:top_k]
