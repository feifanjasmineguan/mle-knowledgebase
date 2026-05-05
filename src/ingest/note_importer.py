# src/ingest/note_importer.py
from pathlib import Path
from config import MAX_ARTICLE_CHARS

def load_note(file_path: str) -> dict:
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")

    return {
        "title": path.stem,
        "content": content[:MAX_ARTICLE_CHARS],
        "source_type": "note",
        "source_url": str(path),
    }