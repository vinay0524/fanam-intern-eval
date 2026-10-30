from app.agent.generator import ask_gemini


def rewrite_query(query: str, history=None) -> str:
    history_text = str(history) if history else "No previous conversation."

    prompt = f"""
    Rewrite the user's question to make it clear and searchable.

    Conversation history:
    {history_text}

    User question:
    {query}

    Return only the rewritten question.
    Do not change its original meaning.
    """

    return ask_gemini(prompt).strip()