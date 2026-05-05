# src/ingest/watcher.py
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import RAW_DIR, WIKI_DIR
from src.ingest.web_clipper import clip_url
from src.ingest.pdf_loader import load_pdf
from src.ingest.note_importer import load_note
from src.compiler.summarizer import summarize_source
from src.compiler.index_builder import build_index
from src.compiler.linker import add_backlinks
from src.embeddings.build_store import build_vector_store

# Track files already processed so we don't re-ingest on restart
PROCESSED_LOG = RAW_DIR / ".processed"

def load_processed() -> set:
    if PROCESSED_LOG.exists():
        return set(PROCESSED_LOG.read_text().splitlines())
    return set()

def mark_processed(filename: str):
    with open(PROCESSED_LOG, "a") as f:
        f.write(filename + "\n")

def process_file(path: Path):
    suffix = path.suffix.lower()
    filename = path.name

    processed = load_processed()
    if filename in processed:
        return  # already handled

    print(f"\n📥 New file detected: {filename}")

    try:
        if suffix == ".pdf":
            source = load_pdf(str(path))
        elif suffix in (".txt", ".md"):
            source = load_note(str(path))
        elif suffix == ".url":
            # A .url file contains a single URL on the first line
            url = path.read_text().strip().splitlines()[0]
            source = clip_url(url)
        else:
            print(f"  Skipping unsupported file type: {suffix}")
            return

        summarize_source(source)
        build_index()
        add_backlinks()
        build_vector_store()
        mark_processed(filename)
        print(f"✓ Done — wiki and vector store updated")

    except Exception as e:
        print(f"  ✗ Failed to process {filename}: {e}")


class RawFolderHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        # Short delay so the file is fully written before we read it
        time.sleep(1)
        process_file(Path(event.src_path))


def start_watcher():
    RAW_DIR.mkdir(exist_ok=True)
    WIKI_DIR.mkdir(exist_ok=True)

    print(f"👀 Watching {RAW_DIR} for new files...")
    print("Drop a .pdf, .txt, .md, or .url file into raw/ to auto-ingest.")
    print("Press Ctrl+C to stop.\n")

    observer = Observer()
    observer.schedule(RawFolderHandler(), str(RAW_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatcher stopped.")
    observer.join()