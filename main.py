import sys
import argparse
import json
import threading
import webbrowser
import uvicorn
from app.models import AgentRequest
from app.agent import FinancialResearchAgent


def run_cli_query(query: str, persona: str = "Junior Analyst", deep_search: bool = False):
    print(f"\n{'='*80}")
    print(f"Executing Query: {query}")
    print(f"Persona: {persona} | Deep Search: {deep_search}")
    print(f"{'='*80}\n")

    agent = FinancialResearchAgent()
    agent.initialize()

    request = AgentRequest(
        session_id="cli_session_001",
        query=query,
        persona=persona,
        deep_search=deep_search,
    )

    response = agent.run(request)
    print("\n--- AGENT RESPONSE JSON ---")
    print(json.dumps(response.model_dump(), indent=2))
    print("\n--- FORMATTED ANSWER ---")
    print(response.answer)
    print(f"{'='*80}\n")


def start_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False):
    print(f"Starting Financial Research Assistant Agent API on http://{host}:{port}")
    if open_browser:
        target_url = f"http://{host}:{port}"
        print(f"Opening browser at {target_url}...")
        threading.Timer(1.5, lambda: webbrowser.open(target_url)).start()
    uvicorn.run("app.server:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="Financial Research Assistant Agentic RAG System")
    parser.add_argument("--server", action="store_true", help="Start FastAPI uvicorn server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--browser", action="store_true", help="Open browser on server startup")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion and build FAISS/BM25 indices")
    parser.add_argument("--query", type=str, help="Run a query directly from CLI")
    parser.add_argument("--persona", type=str, default="Junior Analyst", help="Persona for query")
    parser.add_argument("--deep-search", action="store_true", help="Enable deep search decomposition")

    args = parser.parse_args()

    if args.server:
        start_server(args.host, args.port, open_browser=args.browser)
    elif args.ingest:
        agent = FinancialResearchAgent()
        agent.initialize(force_reindex=True)
        print(f"Ingestion finished. Total chunks indexed: {len(agent.indexer.chunks)}")
    elif args.query:
        run_cli_query(args.query, persona=args.persona, deep_search=args.deep_search)
    else:
        # Default behavior: run ingestion check and a sample query
        print("Running full pipeline validation...")
        agent = FinancialResearchAgent()
        agent.initialize(force_reindex=True)
        print(f"Indexed {len(agent.indexer.chunks)} chunks.")

        sample_queries = [
            ("What was Nimbus Robotics (NMBR) net revenue and gross margin in Q2 2026?", "Junior Analyst", False),
            ("Compare the gross margins and leverage profiles of NMBR and SLDC.", "Portfolio Manager", True),
            ("Why did NMBR's gross margin compress, how did it impact EPS, and why did debt-to-equity rise?", "Compliance Reviewer", True),
        ]

        for q, p, ds in sample_queries:
            req = AgentRequest(
                session_id="validation_run",
                query=q,
                persona=p,
                deep_search=ds,
            )
            res = agent.run(req)
            print(f"\n[Validated Query]: {q}")
            print(f"Query Type: {res.query_type} | Attempts: {res.retrieval_attempts} | Sufficient: {res.retrieval_sufficient} | Confidence: {res.confidence_score}")
            print(f"Used Chunks: {res.used_chunks}")
            print(f"Answer snippet: {res.answer[:200]}...\n")


if __name__ == "__main__":
    main()
