import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import pypdf
import docx


class LoadedDocument:
    def __init__(
        self,
        content: str,
        document_name: str,
        document_type: str,
        page_number: int = 1,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.document_name = document_name
        self.document_type = document_type
        self.page_number = page_number
        self.extra_metadata = extra_metadata or {}

    def __repr__(self) -> str:
        return f"<LoadedDocument name={self.document_name} type={self.document_type} page={self.page_number} len={len(self.content)}>"


class PDFDocumentLoader:
    @staticmethod
    def load(file_path: Path) -> List[LoadedDocument]:
        documents: List[LoadedDocument] = []
        try:
            reader = pypdf.PdfReader(str(file_path))
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                cleaned_text = text.strip()
                if cleaned_text:
                    documents.append(
                        LoadedDocument(
                            content=cleaned_text,
                            document_name=file_path.name,
                            document_type="pdf",
                            page_number=page_idx + 1,
                        )
                    )
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return documents


class DocxDocumentLoader:
    @staticmethod
    def load(file_path: Path) -> List[LoadedDocument]:
        documents: List[LoadedDocument] = []
        try:
            doc = docx.Document(str(file_path))
            full_text_blocks: List[str] = []

            for paragraph in doc.paragraphs:
                p_text = paragraph.text.strip()
                if p_text:
                    full_text_blocks.append(p_text)

            for table in doc.tables:
                table_rows = []
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        table_rows.append(" | ".join(row_cells))
                if table_rows:
                    full_text_blocks.append("\n".join(table_rows))

            if full_text_blocks:
                full_content = "\n\n".join(full_text_blocks)
                documents.append(
                    LoadedDocument(
                        content=full_content,
                        document_name=file_path.name,
                        document_type="docx",
                        page_number=1,
                    )
                )
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
        return documents


class TextDocumentLoader:
    @staticmethod
    def load(file_path: Path) -> List[LoadedDocument]:
        documents: List[LoadedDocument] = []
        content = ""
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        cleaned = content.strip()
        if cleaned:
            documents.append(
                LoadedDocument(
                    content=cleaned,
                    document_name=file_path.name,
                    document_type="txt",
                    page_number=1,
                )
            )
        return documents


class StructuredTabularLoader:
    COLUMN_NAME_MAP = {
        "ticker": "Ticker",
        "quarter": "Financial Quarter",
        "net_revenue_musd": "Net Revenue ($ Millions)",
        "gross_margin_pct": "Gross Margin (%)",
        "operating_income_musd": "Operating Income ($ Millions)",
        "diluted_eps_usd": "Diluted EPS ($)",
        "total_debt_musd": "Total Debt ($ Millions)",
        "total_equity_musd": "Total Equity ($ Millions)",
        "debt_to_equity": "Debt-to-Equity Ratio",
        "current_ratio": "Current Ratio",
        "interest_coverage": "Interest Coverage Ratio",
        "stock_price_usd": "Stock Price ($)",
        "forward_pe": "Forward P/E Ratio",
        "ttm_revenue_growth_pct": "TTM Revenue Growth (%)",
    }

    @classmethod
    def load(cls, file_path: Path) -> List[LoadedDocument]:
        documents: List[LoadedDocument] = []
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".csv":
                df = pd.read_csv(file_path)
            elif suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                return documents

            # Parse each row into structured financial record
            for row_idx, row in df.iterrows():
                row_items = []
                for col in df.columns:
                    val = row[col]
                    friendly_col = cls.COLUMN_NAME_MAP.get(str(col).strip().lower(), str(col))
                    row_items.append(f"{friendly_col}: {val}")

                row_narrative = (
                    f"Financial Statement Record [File: {file_path.name}, Row {row_idx + 1}]:\n"
                    + " | ".join(row_items)
                )

                documents.append(
                    LoadedDocument(
                        content=row_narrative,
                        document_name=file_path.name,
                        document_type=suffix.lstrip("."),
                        page_number=int(row_idx) + 1,
                    )
                )

            # Also create aggregate summaries by ticker for comparative queries
            if "ticker" in df.columns:
                for ticker, group in df.groupby("ticker"):
                    summary_lines = [
                        f"Financial Summary Table for Ticker {ticker} ({file_path.name}):",
                        group.to_string(index=False)
                    ]
                    documents.append(
                        LoadedDocument(
                            content="\n".join(summary_lines),
                            document_name=file_path.name,
                            document_type=suffix.lstrip("."),
                            page_number=1,
                        )
                    )

        except Exception as e:
            print(f"Error loading tabular data from {file_path}: {e}")

        return documents


def load_documents_from_directory(directory_path: Path) -> List[LoadedDocument]:
    all_docs: List[LoadedDocument] = []
    if not directory_path.exists():
        print(f"Directory {directory_path} does not exist.")
        return all_docs

    for file_path in sorted(directory_path.iterdir()):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            all_docs.extend(PDFDocumentLoader.load(file_path))
        elif suffix in [".docx", ".doc"]:
            all_docs.extend(DocxDocumentLoader.load(file_path))
        elif suffix == ".txt":
            all_docs.extend(TextDocumentLoader.load(file_path))
        elif suffix in [".csv", ".xlsx", ".xls"]:
            all_docs.extend(StructuredTabularLoader.load(file_path))

    return all_docs
