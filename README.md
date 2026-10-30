# Financial Document Intelligence RAG (Hybrid Dense + Sparse Retrieval)

An enterprise-grade Financial Document Intelligence and Research Assistant with a modern **React (Vite)** frontend and high-performance **FastAPI (Python)** backend, featuring true **Hybrid Search (Dense + Sparse Retrieval)** powered by **Pinecone Serverless** and **Google Gemini**.

> **Architecture Guarantee**: This project strictly adheres to a clean, pure Python architecture. It does **NOT** use LangChain or LangGraph. All document parsing, sliding-window chunking, dense embeddings, BM25 sparse vector calculation, weighted hybrid retrieval, and LLM inference use direct, transparent Python and official REST APIs / SDKs.

---

## 🏗️ Architecture & Data Flow

```text
              REACT (VITE FRONTEND)
                     │
                     ▼
             FASTAPI (PYTHON BACKEND)
                     │
                     ▼
          MULTI-FORMAT UPLOADER (.docx, .pdf, .csv, .txt, .xlsx)
                     │
                     ▼
         DOCUMENT PARSER (Text & Table Extraction)
                     │
                     ▼
          TEXT CHUNKER (Sliding Window with Overlap)
                     │
                     ▼
         ┌────────────────────────────────┐
         │     HYBRID VECTOR GENERATION   │
         │ 1. Dense: Gemini (3072 dims)   │
         │ 2. Sparse: BM25 Token Weights  │
         └────────────────────────────────┘
                     │
                     ▼
          PINECONE VECTOR DATABASE (Serverless Hybrid Index)
                     │
                     │
                     ▼
             USER FINANCIAL QUERY
                     │
                     ▼
         ┌────────────────────────────────┐
         │     HYBRID QUERY ENCODING      │
         │ 1. Dense: Gemini Query Vector  │
         │ 2. Sparse: BM25 Query Vector   │
         │ 3. Alpha Weighting (0.0 - 1.0) │
         └────────────────────────────────┘
                     │
                     ▼
          HYBRID PINECONE RETRIEVAL (Top-k Excerpts)
                     │
                     ▼
          GOOGLE GEMINI LLM (gemini-3.5-flash-lite)
                     │
                     ▼
         NATURAL ANALYTICAL EXPLANATION WITH CITATIONS
                     │
                     ▼
              REACT (VITE FRONTEND)
```

---

## ✨ Features

1. **Multi-Document Upload**:
   - Simultaneous batch upload supporting `.docx`, `.pdf`, `.csv`, `.txt`, `.xlsx`, and `.xls`.
   - Robust extraction of financial text, narrative notes, earnings call transcripts, and structured tables.

2. **True Hybrid Retrieval (Dense + Sparse)**:
   - **Dense Retrieval**: High-dimensional semantic embeddings via Google Gemini (`models/gemini-embedding-001`, 3072 dims).
   - **Sparse Retrieval**: Pure Python BM25 sparse vector encoder outputting Pinecone-compliant `{"indices": [...], "values": [...]}` for exact financial term, ticker, and numeric figure matching.
   - **Alpha Weighting**: Dynamically balanced hybrid retrieval ($0.0 \le \alpha \le 1.0$).

3. **Pinecone Serverless Hybrid Index**:
   - Manages serverless hybrid indices with `dotproduct` metric.
   - Upserts dense and sparse vector components in unified upsert transactions.

4. **Accurate Financial Synthesis**:
   - Grounded LLM reasoning powered by Google Gemini (`models/gemini-3.5-flash-lite`).
   - Clear explanations focused directly on the user's inquiry with underlying financial drivers (margins, leverage, liquidity, interest coverage).

5. **Enterprise 3-Panel React UI**:
   - **Left Panel**: Documents manager with drag-and-drop upload, file list, and document options menu (**View** details modal & **Delete** document).
   - **Center Panel**: Conversational interface with **Clear Chat** history reset and markdown formatting.
   - **Right Panel**: Retrieved document source excerpts with similarity scores and chunk references.
   - **Theme Toggle**: Real-time Dark Mode and Light Mode switching with local storage persistence.

---

## 🔐 Environment Configuration

All sensitive API keys are kept in local `.env` files and excluded from Git commits via `.gitignore`. Template files (`.env.example`) are provided.

### Backend (`backend/.env`):

```env
# ==========================================
# GEMINI CONFIGURATION
# ==========================================
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_LLM_MODEL=models/gemini-3.5-flash-lite

# ==========================================
# PINECONE VECTOR DATABASE CONFIGURATION
# ==========================================
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=financial-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# ==========================================
# HYBRID RETRIEVAL SETTINGS
# ==========================================
DEFAULT_ALPHA=0.5

# ==========================================
# SERVER CONFIGURATION
# ==========================================
PORT=8000
HOST=0.0.0.0
```

### Frontend (`frontend/.env`):

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 🚀 Setup & Execution Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Pinecone API Key
- Google Gemini API Key

---

### Step 1: Set Up Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Mac/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your `.env` file:
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY and PINECONE_API_KEY in .env
   ```

5. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   The backend API will be live at `http://localhost:8000` (interactive API docs available at `http://localhost:8000/docs`).

---

### Step 2: Set Up Frontend

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node dependencies:
   ```bash
   npm install
   ```

3. Configure your `.env` file:
   ```bash
   cp .env.example .env
   ```

4. Start the Vite development server:
   ```bash
   npm run dev
   ```
   The application will be live at `http://localhost:5173`.

---

## 🧪 Testing Document Ingestion & Retrieval Pipeline

To verify multi-format parsing, sliding-window chunking, and BM25 sparse vector generation:

```bash
cd backend
python test_pipeline.py
```
