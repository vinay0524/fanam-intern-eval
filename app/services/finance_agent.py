from app.agent.classifier import classify_query
from app.agent.rewriter import rewrite_query
from app.agent.decomposer import decompose_query
from app.agent.planner import create_plan
from app.agent.retry import should_retry
from app.agent.generator import generate_answer
from app.agent.verifier import verify_answer
from app.agent.confidence import calculate_confidence

from app.retrieval.dense import dense_search
from app.retrieval.sparse import sparse_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank_results

from app.memory.session_store import load_history, save_history


def retrieve_context(query: str, top_k: int = 5) -> list:
    """Perform hybrid retrieval and reranking."""

    dense_results = dense_search(query, top_k)
    sparse_results = sparse_search(query, top_k)

    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        top_k
    )

    final_results = rerank_results(
        query,
        fused_results,
        top_k
    )

    return final_results


def run_finance_agent(
    query: str,
    session_id: str = "default",
    persona: str = "financial_analyst",
    deep_search: bool = False
) -> dict:

    # 1. Load conversation history
    history = load_history(session_id)

    # 2. Classify the query
    query_type = classify_query(query)

    # 3. Create execution plan
    plan = create_plan(query, query_type)

    # 4. Rewrite query when necessary
    search_query = query

    if plan.get("rewrite", False):
        search_query = rewrite_query(query, history)

    # 5. Decompose complex questions
    if deep_search or plan.get("decompose", False):
        subqueries = decompose_query(search_query)
    else:
        subqueries = [search_query]

    # 6. Hybrid retrieval
    all_results = []

    for subquery in subqueries:
        results = retrieve_context(
            subquery,
            plan.get("top_k", 5)
        )

        all_results.extend(results)

    # 7. Retry if no evidence is retrieved
    attempt = 0
    max_retries = 2

    while should_retry(all_results, attempt, max_retries):
        attempt += 1

        retry_query = rewrite_query(search_query, history)

        all_results = retrieve_context(
            retry_query,
            plan.get("top_k", 5)
        )

    # 8. Remove duplicate chunks
    unique_results = []
    seen = set()

    for result in all_results:
        key = result.get("id", result.get("text", ""))

        if key not in seen:
            seen.add(key)
            unique_results.append(result)

    # 9. Generate answer using Gemini
    answer = generate_answer(
        query=query,
        contexts=unique_results,
        history=history
    )

    # 10. Verify answer
    verification = verify_answer(
        answer,
        unique_results
    )

    # 11. Calculate confidence
    confidence = calculate_confidence(
        unique_results,
        verification
    )

    # 12. Save conversation
    save_history(
        session_id,
        query,
        answer
    )

    # 13. Return final response
    return {
        "session_id": session_id,
        "query": query,
        "query_type": query_type,
        "answer": answer,
        "sources": [
            result.get("metadata", {})
            for result in unique_results
        ],
        "confidence": confidence,
        "verification": verification,
        "retrieval_count": len(unique_results),
        "retry_attempts": attempt
    }
