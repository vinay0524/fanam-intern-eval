import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from app.ingestion.loaders import load_document
from app.ingestion.metadata import extract_metadata
from app.ingestion.chunking import split_text


RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/indexes")
OUTPUT_FILE = INDEX_DIR / "chunks.json"


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for file_path in RAW_DIR.iterdir():
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        try:
            documents = load_document(str(file_path))

            for document in documents:
                metadata = extract_metadata(
                    str(file_path), document
                )

                chunks = split_text(document["text"])

                for index, chunk in enumerate(chunks):
                    all_chunks.append({
                        "id": len(all_chunks),
                        "chunk_index": index,
                        "text": chunk,
                        "metadata": metadata
                    })

            print(f"Processed: {file_path.name}")

        except Exception as error:
            print(f"Failed: {file_path.name}: {error}")

    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


ingest_all_documents = main


if __name__ == "__main__":
    main()
