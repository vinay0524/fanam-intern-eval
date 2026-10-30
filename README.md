Adaptive Agentic RAG Financial Research Assistant (Version 3.2.6)

A production-grade, agentic Retrieval-Augmented Generation (RAG) system designed to answer complex financial investment and market analysis queries from multi-format document collections (PDF, DOCX, TXT, CSV, and Excel).

🚀 Key Features & Architecture

Intelligent Multi-Format Document Ingestion: Automatically loads, parses, and chunks documents (SEC filings, earnings call transcripts, analyst notes, financial spreadsheets) with fine-grained metadata tracking (document_name, document_type, page_number, chunk_id).

Hybrid Retrieval (Dense + Sparse + RRF): Combines dense vector similarity search (ChromaDB) with keyword-based sparse retrieval (BM25), merged mathematically via Reciprocal Rank Fusion (RRF).

Agentic Query Decomposition: Dynamically classifies queries and decomposes complex multi-hop questions into targeted sub-queries for independent retrieval cycles.

Sufficiency Assessment & Adaptive Retries: Evaluates retrieved context quality and triggers bounded automatic retries with expanded query terms if evidence is insufficient.

Dynamic Financial Calculations: Automatically extracts quantitative metrics (such as revenue and operating income) from retrieved evidence to compute financial ratios on the fly.

Persona-Based Synthesis (Live Gemini LLM): Generates tailored responses matching specific user personas (Junior Analyst, Portfolio Manager, or Compliance Reviewer) backed by strict evidence grounding and chunk citations.

Assessment-Compliant 5-Factor Confidence Scoring: Calculates confidence scores using a weighted formula accounting for retrieval similarity, source diversity, sufficiency, retry attempts, and verification success.

🛠️ Project File Structure

financial_research_assistant/
│
├── documents/                # Place your PDF, DOCX, TXT, CSV, and Excel files here
├── vector_db/                # Persistent ChromaDB vector storage directory
├── app.py                    # Main FastAPI application (Version 3.2.6)
├── requirements.txt          # Python package dependencies
├── .env                      # Environment configuration (GEMINI_API_KEY)
└── README.md                 # Project Documentation


📦 Installation & Setup

Clone or Download the Project Directory and navigate into it:

cd financial_research_assistant


Install Dependencies:

python.exe -m pip install fastapi uvicorn pydantic chromadb rank_bm25 pypdf python-docx openpyxl pandas python-dotenv google-genai numpy


Configure your Gemini API Key:
Create a .env file in the project root and add your secure API key:

GEMINI_API_KEY=your_actual_gemini_api_key_here


(Or set it directly in your PowerShell terminal: $env:GEMINI_API_KEY="your_key")

Add Source Documents:
Place your evaluation documents (.pdf, .docx, .txt, .csv, .xlsx) inside the documents/ folder.

🏃 Running the Application

Start the FastAPI server using Uvicorn:

python.exe app.py


The server will initialize the native ChromaDB vector store, build the BM25 sparse index, and run locally on http://localhost:8000.

Interactive API Docs (Swagger UI): Open http://localhost:8000/docs in your browser.

🔌 API Specification

POST /finance-agent

Request Body Example:

{
  "session_id": "fin456",
  "query": "What was the company's revenue and operating margin in the latest reporting period?",
  "persona": "Junior Analyst",
  "deep_search": true
}


Response Body Example:

{
  "session_id": "fin456",
  "query_type": "Multi-Hop",
  "original_query": "What was the company's revenue and operating margin in the latest reporting period?",
  "rewritten_query": "Optimized financial search query...",
  "sub_queries": [
    "Primary operational drivers and cost factors...",
    "Subsequent financial results and profitability impact..."
  ],
  "deep_search_used": true,
  "retrieval_attempts": 1,
  "retrieval_sufficient": true,
  "confidence_score": 0.96,
  "sources": [
    {
      "document_name": "Nimbus_Robotics_10Q_Q2_2026.pdf",
      "document_type": "PDF",
      "page_number": 1,
      "chunk_id": "chunk_16"
    }
  ],
  "retrieval_metrics": {
    "average_similarity": 0.9,
    "source_diversity": 1.0,
    "evidence_coverage": 1.0
  },
  "verification": {
    "faithfulness_score": 0.95,
    "unsupported_claims": [],
    "market_data_available": true
  },
  "answer": "Based on retrieved evidence..."
}