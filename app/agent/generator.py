import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=API_KEY)


def ask_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text or ""


def generate_answer(query: str, contexts: list, history=None) -> str:
    context_text = "\n\n".join(
        str(item.get("text", item)) if isinstance(item, dict) else str(item)
        for item in contexts
    )

    prompt = f"""
    You are an expert financial research assistant.

    Answer the user's question using only the provided
    financial document context.

    Conversation history:
    {history or "No previous conversation."}

    Financial document context:
    {context_text}

    User question:
    {query}

    Instructions:
    1. Give a clear and accurate answer.
    2. Do not invent financial facts.
    3. If the information is unavailable, say so.
    4. Mention the source document and page when available.
    """

    return ask_gemini(prompt)
