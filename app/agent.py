import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.config import (
    DATA_DIR,
    INDEX_DIR,
    QUERY_MULTI_HOP,
    QUERY_DECISION_SUPPORT,
)
from app.models import (
    AgentRequest,
    AgentResponse,
    DocumentChunk,
    SufficiencyResult,
    VerificationResult,
    ConfidenceBreakdown,
)
from app.ingestion.loaders import load_documents_from_directory
from app.ingestion.chunker import FinancialDocumentChunker
from app.ingestion.indexer import HybridIndexer
from app.retrieval.classifier import QueryClassifier
from app.retrieval.rewriter import QueryRewriter
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.sufficiency import RetrievalSufficiencyAssessor
from app.retrieval.retry_handler import RetryHandler
from app.generation.generator import PersonaResponseGenerator
from app.generation.verifier import FaithfulnessVerifier
from app.generation.confidence import ConfidenceScorer

logger = logging.getLogger("finance_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FinancialResearchAgent:
    """
    Production-grade Agentic RAG system executing the dual-phase ingestion & retrieval pipeline.
    """

    def __init__(self, data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR):
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.indexer = HybridIndexer(index_dir=index_dir)
        self.searcher = HybridSearcher(self.indexer)
        self.chunker = FinancialDocumentChunker()
        self.is_initialized = False

    def initialize(self, force_reindex: bool = False) -> None:
        """
        Loads persisted indices or runs document ingestion pipeline.
        """
        if not force_reindex and self.indexer.load_indices():
            logger.info("Successfully loaded existing vector and BM25 indices.")
            self.is_initialized = True
            return

        logger.info(f"Ingesting documents from {self.data_dir}...")
        raw_docs = load_documents_from_directory(self.data_dir)
        logger.info(f"Loaded {len(raw_docs)} document nodes from {self.data_dir}.")

        chunks = self.chunker.chunk_documents(raw_docs)
        logger.info(f"Created {len(chunks)} chunks with metadata.")

        self.indexer.build_indices(chunks)
        self.indexer.save_indices()
        self.is_initialized = True
        logger.info("Ingestion and indexing complete.")

    def run(self, request: AgentRequest) -> AgentResponse:
        """
        Executes Query & Retrieval Execution Pipeline.
        """
        if not self.is_initialized:
            self.initialize()

        session_id = request.session_id
        original_query = request.query
        persona = request.persona
        force_deep_search = request.deep_search

        # 1. Query Understanding
        query_type = QueryClassifier.classify(original_query)
        logger.info(f"[{session_id}] Query Classified as: '{query_type}'")

        # 2. Query Rewriting & Decomposition
        rewritten_query = QueryRewriter.rewrite_query(original_query)
        logger.info(f"[{session_id}] Rewritten Query: '{rewritten_query}'")

        should_deep_search = force_deep_search or (query_type in [QUERY_MULTI_HOP, QUERY_DECISION_SUPPORT])
        if should_deep_search:
            sub_queries = QueryRewriter.decompose_query(original_query, query_type)
            search_queries = sub_queries
            deep_search_used = True
            logger.info(f"[{session_id}] Deep Search Activated. Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        else:
            search_queries = [rewritten_query]
            deep_search_used = False

        # 3. Hybrid Retrieval & Retry Execution
        retrieval_result = RetryHandler.execute_with_retries(
            searcher=self.searcher,
            initial_queries=search_queries,
            original_query=original_query,
            max_retries=2,
        )

        attempts = retrieval_result["attempts"]
        retrieved_chunks: List[DocumentChunk] = retrieval_result["chunks"]
        sufficiency: SufficiencyResult = retrieval_result["sufficiency"]
        logger.info(f"[{session_id}] Retrieval completed in {attempts} attempt(s). Sufficient: {sufficiency.is_sufficient} ({sufficiency.reason})")

        # 4. Persona-Based Response Generation
        raw_answer, used_chunks = PersonaResponseGenerator.generate(
            query=original_query,
            query_type=query_type,
            persona=persona,
            chunks=retrieved_chunks,
        )

        # 5. Faithfulness Verification
        verification = FaithfulnessVerifier.verify(raw_answer, retrieved_chunks)
        final_answer = verification.cleaned_answer or raw_answer
        logger.info(f"[{session_id}] Faithfulness verification success rate: {verification.success_rate:.2%}")

        # 6. Weighted Confidence Scoring
        confidence: ConfidenceBreakdown = ConfidenceScorer.compute_confidence(
            chunks=retrieved_chunks,
            sufficiency=sufficiency,
            verification=verification,
            attempts=attempts,
        )
        logger.info(f"[{session_id}] Computed confidence score: {confidence.total_confidence} (Breakdown: {confidence.model_dump()})")

        return AgentResponse(
            session_id=session_id,
            query_type=query_type,
            original_query=original_query,
            rewritten_query=rewritten_query,
            deep_search_used=deep_search_used,
            retrieval_attempts=attempts,
            retrieval_sufficient=sufficiency.is_sufficient,
            confidence_score=confidence.total_confidence,
            used_chunks=used_chunks,
            answer=final_answer,
        )

