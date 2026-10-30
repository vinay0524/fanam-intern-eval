
from pathlib import Path
import fitz
import pandas as pd
from docx import Document


SUPPORTED_FORMATS = {
    ".pdf", ".docx", ".txt", ".csv", ".xlsx"
}


def load_document(file_path: str) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {extension}")

    documents = []

    if extension == ".pdf":
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                documents.append({
                    "text": page.get_text(),
                    "page": page_number
                })

    elif extension == ".docx":
        doc = Document(path)
        text = "\n".join(
            paragraph.text for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )
        documents.append({"text": text, "page": None})

    elif extension == ".txt":
        text = path.read_text(encoding="utf-8")
        documents.append({"text": text, "page": None})

    elif extension == ".csv":
        df = pd.read_csv(path)
        documents.append({
            "text": df.to_string(index=False),
            "page": None
        })

    elif extension == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None)

        for sheet_name, df in sheets.items():
            documents.append({
                "text": df.to_string(index=False),
                "page": None,
                "sheet": sheet_name
            })

    return documents
