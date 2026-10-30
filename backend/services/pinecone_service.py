import time
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from config import settings

class PineconeService:
    def __init__(self):
        self._pc = None
        self._cached_key = None
        self._index = None

    @property
    def pc(self) -> Pinecone:
        current_key = settings.PINECONE_API_KEY
        if not self._pc or self._cached_key != current_key:
            if not current_key or "your_" in current_key:
                raise ValueError("PINECONE_API_KEY is not configured in backend/.env file.")
            self._pc = Pinecone(api_key=current_key)
            self._cached_key = current_key
            self._index = None
        return self._pc

    def get_or_create_index(self, dimension: int = 768):
        """
        Ensures Pinecone index exists with matching dimension and 'dotproduct' metric.
        """
        if self._index:
            return self._index

        index_name = settings.PINECONE_INDEX_NAME
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if index_name in existing_indexes:
            desc = self.pc.describe_index(index_name)
            # If dimension changed (e.g. switching from OpenAI 1536 to Gemini 768)
            if desc.dimension != dimension:
                print(f"Index '{index_name}' dimension mismatch ({desc.dimension} != {dimension}). Re-creating index...")
                self.pc.delete_index(index_name)
                while index_name in [idx["name"] for idx in self.pc.list_indexes()]:
                    time.sleep(1)
                existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            print(f"Creating Pinecone index '{index_name}' (dimension={dimension}, metric='dotproduct')...")
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION
                )
            )
            while not self.pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        self._index = self.pc.Index(index_name)
        return self._index

    def upsert_hybrid_chunks(
        self,
        chunks: List[Dict[str, Any]],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, Any]],
        batch_size: int = 50
    ) -> int:
        """
        Upserts chunk records with both dense vectors and sparse vectors for hybrid retrieval.
        """
        index = self.get_or_create_index(dimension=len(dense_vectors[0]))
        total = len(chunks)
        upserted_count = 0

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_dense = dense_vectors[i:i + batch_size]
            batch_sparse = sparse_vectors[i:i + batch_size]

            vectors_to_upsert = []
            for chunk, dense, sparse in zip(batch_chunks, batch_dense, batch_sparse):
                vector_data = {
                    "id": chunk["chunk_id"],
                    "values": dense,
                    "metadata": {
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "char_count": chunk["char_count"]
                    }
                }
                if sparse.get("indices") and sparse.get("values"):
                    vector_data["sparse_values"] = sparse
                
                vectors_to_upsert.append(vector_data)

            index.upsert(vectors=vectors_to_upsert)
            upserted_count += len(vectors_to_upsert)

        return upserted_count

    def hybrid_query(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[str, Any],
        top_k: int = 5,
        alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a hybrid search query combining dense and sparse vectors:
        - alpha = 1.0 -> 100% dense (pure semantic)
        - alpha = 0.0 -> 100% sparse (pure keyword/BM25)
        - alpha = 0.5 -> balanced hybrid search
        """
        if alpha is None:
            alpha = settings.DEFAULT_ALPHA
        
        alpha = max(0.0, min(1.0, float(alpha)))
        index = self.get_or_create_index(dimension=len(dense_vector))

        scaled_dense = [val * alpha for val in dense_vector]

        scaled_sparse = None
        if sparse_vector.get("indices") and sparse_vector.get("values"):
            scaled_sparse = {
                "indices": sparse_vector["indices"],
                "values": [round(val * (1.0 - alpha), 5) for val in sparse_vector["values"]]
            }

        query_kwargs = {
            "top_k": top_k,
            "include_metadata": True
        }

        if alpha > 0.0:
            query_kwargs["vector"] = scaled_dense
        if alpha < 1.0 and scaled_sparse:
            query_kwargs["sparse_vector"] = scaled_sparse

        result = index.query(**query_kwargs)

        retrieved = []
        for match in result.get("matches", []):
            retrieved.append({
                "id": match["id"],
                "score": round(match.get("score", 0.0), 4),
                "text": match.get("metadata", {}).get("text", ""),
                "source": match.get("metadata", {}).get("source", "unknown"),
                "chunk_index": match.get("metadata", {}).get("chunk_index", 0)
            })

        return retrieved
