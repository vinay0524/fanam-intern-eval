#!/usr/bin/env python3
"""
Financial Research Assistant Agentic RAG System
Automated Evaluation Benchmark Runner
Executes 7 official benchmark queries against the live FastAPI endpoint (POST /finance-agent)
and validates contract compliance, retrieval sufficiency, and persona synthesis.
"""

import sys
import json
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any

API_ENDPOINT = "http://127.0.0.1:8000/finance-agent"

BENCHMARK_TEST_CASES = [
    {
        "id": "Q1",
        "category": "Multi-Hop / Fact Retrieval",
        "query": "How did rising input costs affect gross margin, and how did that influence quarterly earnings per share?",
        "persona": "Junior Analyst",
        "deep_search": True,
    },
    {
        "id": "Q2",
        "category": "Comparison / Balance Sheet",
        "query": "Compare the debt-to-equity ratios of Company X and Company Y. Why might combining leverage and liquidity metrics improve risk assessment?",
        "persona": "Portfolio Manager",
        "deep_search": True,
    },
    {
        "id": "Q3",
        "category": "Valuation & Growth Trade-off",
        "query": "Does higher revenue growth always translate to higher stock valuation? Justify your answer.",
        "persona": "Portfolio Manager",
        "deep_search": False,
    },
    {
        "id": "Q4",
        "category": "Grounding & Verification",
        "query": "If retrieval similarity scores are low but the generated answer appears correct, should confidence remain high?",
        "persona": "Compliance Reviewer",
        "deep_search": False,
    },
    {
        "id": "Q5",
        "category": "System Resilience",
        "query": "Under what technical conditions should a financial research RAG system trigger retry logic?",
        "persona": "Junior Analyst",
        "deep_search": False,
    },
    {
        "id": "Q6",
        "category": "Verification Architecture",
        "query": "Compare outcome-based, process-based, and hybrid verification approaches. Which is most scalable for financial reporting?",
        "persona": "Compliance Reviewer",
        "deep_search": False,
    },
    {
        "id": "Q7",
        "category": "Infrastructure Trade-offs",
        "query": "A boutique investment firm has limited infrastructure and budget. Should it choose FAISS or ChromaDB? Justify your recommendation.",
        "persona": "Portfolio Manager",
        "deep_search": False,
    },
]

REQUIRED_KEYS = [
    "session_id",
    "query_type",
    "original_query",
    "rewritten_query",
    "deep_search_used",
    "retrieval_attempts",
    "retrieval_sufficient",
    "confidence_score",
    "used_chunks",
    "answer",
]


def send_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    json_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_ENDPOINT,
        data=json_data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        status_code = response.getcode()
        body = response.read().decode("utf-8")
        data = json.loads(body)
        data["_status_code"] = status_code
        return data


def validate_contract(data: Dict[str, Any], test_case: Dict[str, Any]) -> List[str]:
    errors = []
    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"Missing required key: '{key}'")

    if not isinstance(data.get("session_id"), str) or not data.get("session_id"):
        errors.append("Invalid 'session_id': must be non-empty string")

    if not isinstance(data.get("query_type"), str):
        errors.append("Invalid 'query_type': must be string")

    if not isinstance(data.get("confidence_score"), (int, float)) or not (0.0 <= data.get("confidence_score", -1) <= 1.0):
        errors.append(f"Invalid 'confidence_score': {data.get('confidence_score')}, must be float in [0.0, 1.0]")

    if not isinstance(data.get("used_chunks"), list) or len(data.get("used_chunks", [])) == 0:
        errors.append("Invalid 'used_chunks': must be non-empty list of chunk IDs")

    if not isinstance(data.get("answer"), str) or len(data.get("answer", "").strip()) == 0:
        errors.append("Invalid 'answer': must be non-empty string")

    return errors


def main():
    print("=" * 100)
    print("FINANCIAL RESEARCH ASSISTANT AGENTIC RAG — EVALUATION BENCHMARK SUITE")
    print(f"Target API Endpoint: {API_ENDPOINT}")
    print(f"Total Test Cases: {len(BENCHMARK_TEST_CASES)}")
    print("=" * 100)

    results: List[Dict[str, Any]] = []
    all_passed = True

    start_time = time.time()

    for idx, tc in enumerate(BENCHMARK_TEST_CASES, 1):
        q_id = tc["id"]
        category = tc["category"]
        query = tc["query"]
        persona = tc["persona"]
        deep_search = tc["deep_search"]

        print(f"\n[{idx}/7] Running Test Case {q_id} — {category}")
        print(f"    Persona: {persona} | Deep Search: {deep_search}")
        print(f"    Query: \"{query}\"")

        payload = {
            "session_id": f"benchmark_{q_id.lower()}_{int(time.time())}",
            "query": query,
            "persona": persona,
            "deep_search": deep_search,
        }

        try:
            t0 = time.time()
            response_data = send_request(payload)
            latency = time.time() - t0

            status_code = response_data.pop("_status_code", 200)
            validation_errors = validate_contract(response_data, tc)

            if validation_errors:
                all_passed = False
                print(f"    [FAIL] Status: {status_code} | Schema Validation Errors: {validation_errors}")
            else:
                print(f"    [PASS] Status: {status_code} ({latency:.2f}s)")
                print(f"           Query Type:       {response_data['query_type']}")
                print(f"           Rewritten Query:  \"{response_data['rewritten_query']}\"")
                print(f"           Deep Search Used: {response_data['deep_search_used']}")
                print(f"           Attempts:         {response_data['retrieval_attempts']} | Sufficient: {response_data['retrieval_sufficient']}")
                print(f"           Confidence Score: {response_data['confidence_score']:.2f}")
                print(f"           Used Chunks ({len(response_data['used_chunks'])}): {response_data['used_chunks']}")
                answer_preview = response_data['answer'][:160].replace('\n', ' ')
                print(f"           Answer Preview:   {answer_preview}...")

            # Store in output results
            record = {
                "benchmark_id": q_id,
                "category": category,
                "request_payload": payload,
                "latency_seconds": round(latency, 2),
                "contract_valid": len(validation_errors) == 0,
                "validation_errors": validation_errors,
                "response": response_data,
            }
            results.append(record)

        except urllib.error.URLError as e:
            all_passed = False
            print(f"    [ERROR] Connection failed to {API_ENDPOINT}: {e}")
            results.append({
                "benchmark_id": q_id,
                "category": category,
                "request_payload": payload,
                "contract_valid": False,
                "error": str(e),
            })
        except Exception as e:
            all_passed = False
            print(f"    [ERROR] Unexpected execution failure: {e}")
            results.append({
                "benchmark_id": q_id,
                "category": category,
                "request_payload": payload,
                "contract_valid": False,
                "error": str(e),
            })

    total_duration = time.time() - start_time

    # Save to benchmark_results.json
    output_file = "benchmark_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("BENCHMARK EXECUTION SUMMARY")
    print(f"Total Executed: {len(results)} / {len(BENCHMARK_TEST_CASES)}")
    passed_count = sum(1 for r in results if r.get("contract_valid"))
    print(f"Valid Contracts: {passed_count} / {len(BENCHMARK_TEST_CASES)}")
    print(f"Total Execution Time: {total_duration:.2f} seconds")
    print(f"Full Benchmark Payloads Saved to: {output_file}")
    print("=" * 100)

    if not all_passed:
        print("[WARNING] One or more benchmark queries failed validation!")
        sys.exit(1)
    else:
        print("[SUCCESS] All 7 benchmark queries validated successfully against production contract.")
        sys.exit(0)


if __name__ == "__main__":
    main()

