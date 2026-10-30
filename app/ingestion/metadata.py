from pathlib import Path
from uuid import uuid4


def extract_metadata(file_path: str, document: dict) -> dict:
    path = Path(file_path)

    return {
        "chunk_source_id": str(uuid4()),
        "source": path.name,
        "file_type": path.suffix.lower(),
        "page": document.get("page"),
        "sheet": document.get("sheet"),
        "file_path": str(path)
    }
