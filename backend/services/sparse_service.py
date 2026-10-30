import math
import re
import zlib
from collections import Counter
from typing import Dict, List, Any

# Common English stopwords to avoid sparse vector noise
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", 
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

class SparseBM25Service:
    """
    Sparse retrieval vector generator producing {"indices": List[int], "values": List[float]}
    compliant with Pinecone's sparse vector format for hybrid search.
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.avgdl = 100.0  # Default average document length

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
        return [w for w in words if w not in STOP_WORDS and len(w) > 1]

    def _hash_token(self, token: str) -> int:
        """Deterministically map a token string to a non-negative 32-bit integer index for Pinecone."""
        return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF

    def encode_document(self, text: str) -> Dict[str, Any]:
        """
        Encodes a document chunk into a sparse vector dictionary:
        { "indices": [...], "values": [...] }
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        doc_len = len(tokens)
        counts = Counter(tokens)

        indices = []
        values = []

        for token, count in counts.items():
            idx = self._hash_token(token)
            # BM25 Term Frequency saturation formula
            tf = (count * (self.k1 + 1)) / (count + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)))
            indices.append(idx)
            values.append(round(float(tf), 4))

        # Sort by indices as expected by sparse vector indices
        sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
        return {
            "indices": [p[0] for p in sorted_pairs],
            "values": [p[1] for p in sorted_pairs]
        }

    def encode_documents(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Encodes multiple document chunks into sparse vectors."""
        # Update avgdl dynamically based on batch
        total_tokens = sum(len(self._tokenize(t)) for t in texts)
        if texts and total_tokens > 0:
            self.avgdl = max(10.0, total_tokens / len(texts))
        return [self.encode_document(t) for t in texts]

    def encode_query(self, query: str) -> Dict[str, Any]:
        """
        Encodes a user query into a sparse vector dictionary:
        { "indices": [...], "values": [...] }
        """
        tokens = self._tokenize(query)
        if not tokens:
            return {"indices": [], "values": []}

        counts = Counter(tokens)
        indices = []
        values = []

        for token, count in counts.items():
            idx = self._hash_token(token)
            # Query term weighting: 1 + log(count)
            val = 1.0 + math.log(count) if count > 0 else 1.0
            indices.append(idx)
            values.append(round(float(val), 4))

        sorted_pairs = sorted(zip(indices, values), key=lambda x: x[0])
        return {
            "indices": [p[0] for p in sorted_pairs],
            "values": [p[1] for p in sorted_pairs]
        }

