import io
from docx import Document

class DocxParser:
    @staticmethod
    def parse_docx_bytes(file_bytes: bytes, filename: str = "document.docx") -> str:
        """
        Parses a .docx file from bytes and extracts all textual content,
        including headings, body paragraphs, and structured table data.
        """
        doc = Document(io.BytesIO(file_bytes))
        full_text_parts = []

        # 1. Extract paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                full_text_parts.append(text)

        # 2. Extract tables (critical for financial documents)
        for table_idx, table in enumerate(doc.tables):
            table_rows = []
            for row in table.rows:
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                # Avoid duplicate cells from merged rows/columns
                dedup_cells = []
                for cell in row_cells:
                    if not dedup_cells or cell != dedup_cells[-1]:
                        dedup_cells.append(cell)
                if any(dedup_cells):
                    table_rows.append(" | ".join(dedup_cells))
            
            if table_rows:
                table_representation = f"\n[Table {table_idx + 1}]\n" + "\n".join(table_rows) + "\n"
                full_text_parts.append(table_representation)

        return "\n\n".join(full_text_parts)

