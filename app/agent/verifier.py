def verify_answer(answer: str, contexts: list) -> dict:
    if not contexts:
        return {
            "verified": False,
            "message": "No supporting documents were retrieved."
        }

    context_text = " ".join(
        str(item.get("text", item)) if isinstance(item, dict) else str(item)
        for item in contexts
    ).lower()

    answer_words = {
        word.strip(".,!?;:()").lower()
        for word in answer.split()
        if len(word) > 4
    }

    context_words = set(context_text.split())

    supported_words = answer_words.intersection(context_words)

    support_ratio = (
        len(supported_words) / len(answer_words)
        if answer_words else 0
    )

    return {
        "verified": support_ratio >= 0.15,
        "support_ratio": round(support_ratio, 2),
        "message": (
            "Answer has supporting text overlap."
            if support_ratio >= 0.15
            else "Answer needs further evidence checking."
        )
    }