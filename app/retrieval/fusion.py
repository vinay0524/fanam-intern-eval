
def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    top_k: int = 5,
    rrf_k: int = 60
) -> list:

    fused_scores = {}
    documents = {}

    result_lists = [dense_results, sparse_results]

    for results in result_lists:
        for rank, result in enumerate(results, start=1):

            doc_id = result["id"]

            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                documents[doc_id] = result

            fused_scores[doc_id] += 1.0 / (rrf_k + rank)

    ranked_ids = sorted(
        fused_scores,
        key=lambda doc_id: fused_scores[doc_id],
        reverse=True
    )

    fused_results = []

    for doc_id in ranked_ids[:top_k]:
        result = documents[doc_id].copy()
        result["rrf_score"] = fused_scores[doc_id]
        fused_results.append(result)

    return fused_results
