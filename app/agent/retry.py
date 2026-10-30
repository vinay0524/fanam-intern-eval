def should_retry(results: list, attempt: int, max_retries: int = 2) -> bool:
    if attempt >= max_retries:
        return False

    if not results:
        return True

    return False