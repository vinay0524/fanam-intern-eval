import requests
from typing import List
from config import settings

class DenseEmbeddingService:
    def __init__(self):
        pass

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        api_key = settings.GEMINI_API_KEY
        if not api_key or "your_" in api_key:
            raise ValueError("GEMINI_API_KEY is not configured in backend/.env file.")

        model = settings.GEMINI_EMBEDDING_MODEL
        if not model.startswith("models/"):
            model = f"models/{model}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents?key={api_key}"
        all_embeddings = []
        batch_size = 20

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            req_data = {
                "requests": [
                    {
                        "model": model,
                        "content": {"parts": [{"text": t}]},
                        "task_type": "RETRIEVAL_DOCUMENT"
                    }
                    for t in batch
                ]
            }
            res = requests.post(url, json=req_data, timeout=35)
            if not res.ok:
                raise ValueError(f"Gemini Embedding API error ({res.status_code}): {res.text}")

            data = res.json()
            for item in data.get("embeddings", []):
                all_embeddings.append(item.get("values", []))

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        api_key = settings.GEMINI_API_KEY
        if not api_key or "your_" in api_key:
            raise ValueError("GEMINI_API_KEY is not configured in backend/.env file.")

        model = settings.GEMINI_EMBEDDING_MODEL
        if not model.startswith("models/"):
            model = f"models/{model}"

        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={api_key}"
        req_data = {
            "content": {"parts": [{"text": query}]},
            "task_type": "RETRIEVAL_QUERY"
        }
        res = requests.post(url, json=req_data, timeout=35)
        if not res.ok:
            raise ValueError(f"Gemini Query Embedding API error ({res.status_code}): {res.text}")

        return res.json().get("embedding", {}).get("values", [])
