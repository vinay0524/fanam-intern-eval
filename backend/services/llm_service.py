import requests
from typing import List, Dict, Any
from config import settings

class LLMService:
    def __init__(self):
        pass

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesizes an accurate financial response using Google Gemini LLM based on
        retrieved hybrid (Dense + Sparse) context chunks from multi-format documents.
        """
        api_key = settings.GEMINI_API_KEY
        if not api_key or "your_" in api_key:
            raise ValueError("GEMINI_API_KEY is not configured in backend/.env file.")

        model = settings.GEMINI_LLM_MODEL
        if not model.startswith("models/"):
            model = f"models/{model}"

        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("source", "document")
            chunk_idx = chunk.get("chunk_index", i)
            score = chunk.get("score", 0.0)
            text_body = chunk.get("text", "").strip()
            context_parts.append(
                f"[Source Document: {source} | Chunk #{chunk_idx + 1} | Relevance: {score}]\n{text_body}"
            )
        
        context_block = "\n\n" + ("=" * 40) + "\n\n".join(context_parts) if context_parts else "No relevant context found."

        system_instruction = (
            "You are an expert senior financial analyst and research advisor. "
            "Your objective is to provide a comprehensive, well-reasoned, and accurate explanation based STRICTLY on the document excerpts provided below.\n\n"
            "Guidelines:\n"
            "1. Answer ONLY what the user asks, explaining the concepts clearly in full, articulate sentences.\n"
            "2. Do NOT use robotic boilerplate headers like '**Direct Answer:**' or repetitive 'Key details:'. Begin directly with your clear explanation.\n"
            "3. Synthesize the findings into coherent analytical reasoning rather than dumping raw chunk fragments.\n"
            "4. Support your explanation with concrete financial numbers, metrics, ratios, and percentages from the excerpts whenever available.\n"
            "5. If comparing entities or evaluating a claim (e.g. revenue growth vs. valuation), clearly explain the underlying drivers (margins, leverage, risk multiples).\n"
            "6. If the excerpts do not contain sufficient evidence to answer, state that clearly and objectively without speculating."
        )

        full_prompt = (
            f"{system_instruction}\n\n"
            f"DOCUMENT CONTEXT EXCERPTS:\n{context_block}\n\n"
            f"USER QUERY: {query}\n\n"
            f"EXPLANATION:"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent?key={api_key}"
        req_data = {
            "contents": [
                {
                    "parts": [{"text": full_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.15
            }
        }

        res = requests.post(url, json=req_data, timeout=45)
        if not res.ok:
            raise ValueError(f"Gemini LLM API error ({res.status_code}): {res.text}")

        data = res.json()
        try:
            answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            answer_text = "Unable to generate an explanation from Gemini response."

        return {
            "answer": answer_text.strip(),
            "model": model.replace("models/", ""),
            "retrieved_chunks_count": len(context_chunks)
        }
