from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from app.config import RRF_K, TOP_K_DENSE, TOP_K_SPARSE, FINAL_TOP_K
from app.models import DocumentChunk
from app.ingestion.indexer import HybridIndexer


class HybridSearcher:
    def __init__(self, indexer: HybridIndexer, rrf_k: int = RRF_K):
        self.indexer = indexer
        self.rrf_k = rrf_k

    def search_single_query(
        self,
        query: str,
        top_k_dense: int = TOP_K_DENSE,
        top_k_sparse: int = TOP_K_SPARSE,
    ) -> List[DocumentChunk]:
        """
        Executes dense FAISS + sparse BM25 and combines using Reciprocal Rank Fusion (RRF) with constant k=60.
        """
        dense_results = self.indexer.search_dense(query, top_k=top_k_dense)
        sparse_results = self.indexer.search_sparse(query, top_k=top_k_sparse)

        rrf_scores: Dict[str, float] = defaultdict(float)
        dense_similarities: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # 1. Score Dense Ranks (1-based rank)
        for rank_0, (chunk, sim_score) in enumerate(dense_results):
            rank = rank_0 + 1
            chunk_map[chunk.chunk_id] = chunk
            dense_similarities[chunk.chunk_id] = sim_score
            rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        # 2. Score Sparse Ranks (1-based rank)
        for rank_0, (chunk, bm25_score) in enumerate(sparse_results):
            rank = rank_0 + 1
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        # Sort chunks by RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        scored_chunks: List[DocumentChunk] = []
        for cid in sorted_chunk_ids:
            chunk = chunk_map[cid]
            # Attach RRF score and dense similarity
            chunk_copy = DocumentChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=rrf_scores[cid],
            )
            # Store dense similarity as extra attribute for confidence calculations
            chunk_copy.metadata.__dict__["dense_similarity"] = dense_similarities.get(cid, 0.0)
            scored_chunks.append(chunk_copy)

        return scored_chunks

    def search_multi_queries(
        self,
        queries: List[str],
        final_top_k: int = FINAL_TOP_K,
    ) -> List[DocumentChunk]:
        """
        Executes hybrid retrieval across multiple sub-queries with deduplication and score fusion.
        """
        aggregated_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, DocumentChunk] = {}
        best_dense_sim: Dict[str, float] = defaultdict(float)

        for query in queries:
            sub_results = self.search_single_query(query)
            for chunk in sub_results:
                cid = chunk.chunk_id
                chunk_map[cid] = chunk
                score = chunk.score or 0.0
                aggregated_scores[cid] += score
                sim = chunk.metadata.__dict__.get("dense_similarity", 0.0)
                if sim > best_dense_sim[cid]:
                    best_dense_sim[cid] = sim

        sorted_cids = sorted(aggregated_scores.keys(), key=lambda cid: aggregated_scores[cid], reverse=True)

        deduplicated_results: List[DocumentChunk] = []
        for cid in sorted_cids[:final_top_k]:
            orig = chunk_map[cid]
            chunk_copy = DocumentChunk(
                chunk_id=orig.chunk_id,
                content=orig.content,
                metadata=orig.metadata,
                score=aggregated_scores[cid],
            )
            chunk_copy.metadata.__dict__["dense_similarity"] = best_dense_sim[cid]
            deduplicated_results.append(chunk_copy)

        return deduplicated_results

