# src/ingest/pdf_loader.py
import fitz  # pymupdf
from pathlib import Path
from config import MAX_ARTICLE_CHARS

def load_pdf(file_path: str) -> dict:
    path = Path(file_path)
    doc = fitz.open(file_path)

    pages = []
    for page in doc:
        pages.append(page.get_text())

    content = "\n".join(pages)

    return {
        "title": path.stem,
        "content": content[:MAX_ARTICLE_CHARS],
        "source_type": "pdf",
        "source_url": str(path),
    }