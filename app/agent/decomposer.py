from app.agent.generator import ask_gemini


def decompose_query(query: str) -> list:
    prompt = f"""
    Break this financial research question into smaller
    independent questions that can be answered using documents.

    Question: {query}

    Return each subquestion on a separate line.
    Do not add numbering or explanations.
    """

    response = ask_gemini(prompt)

    questions = [
        line.strip("- ").strip()
        for line in response.splitlines()
        if line.strip()
    ]

    return questions[:5] or [query]