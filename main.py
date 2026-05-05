# main.py
import sys
from src.ingest.web_clipper import clip_url
from src.ingest.pdf_loader import load_pdf
from src.ingest.note_importer import load_note
from src.compiler.summarizer import summarize_source
from src.compiler.index_builder import build_index
from src.compiler.linker import add_backlinks
from src.q_and_a.query_handler import ask_wiki

def print_help():
    print("""
ML Knowledge Base CLI
─────────────────────
Ingest
  python main.py ingest:url   <url>        Ingest a web article
  python main.py ingest:pdf   <file.pdf>   Ingest a PDF file
  python main.py ingest:note  <file.md>    Ingest a text/markdown note
  python main.py ingest:batch <urls.txt>   Ingest all URLs from a file

Build
  python main.py compile                   Rebuild index and backlinks
  python main.py build:store               Build semantic search vector store

Query
  python main.py ask          <question>   Ask a question against the wiki
  python main.py flashcards   [csv|md|json] Export Anki flashcards

Automation
  python main.py watch                     Watch raw/ folder for new files
""")

def main():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "ingest:url":
        source = clip_url(args[0])
        summarize_source(source)

    elif command == "ingest:pdf":
        source = load_pdf(args[0])
        summarize_source(source)

    elif command == "ingest:note":
        source = load_note(args[0])
        summarize_source(source)

    elif command == "ingest:batch":
        urls_file = args[0] if args else "urls.txt"
        with open(urls_file) as f:
            urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        print(f"Ingesting {len(urls)} URLs...")
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] {url}")
            try:
                source = clip_url(url)
                summarize_source(source)
            except Exception as e:
                print(f"  ✗ Failed: {e}")

    elif command == "compile":
        build_index()
        add_backlinks()

    elif command == "build:store":
        from src.embeddings.build_store import build_vector_store
        build_vector_store()

    elif command == "ask":
        question = " ".join(args)
        answer = ask_wiki(question)
        print("\n" + answer)

    elif command == "flashcards":
        from src.q_and_a.flashcard_exporter import export_flashcards
        fmt = args[0] if args else "csv"
        export_flashcards(output_format=fmt)

    elif command == "watch":
        from src.ingest.watcher import start_watcher
        start_watcher()

    else:
        print(f"Unknown command: {command}")
        print_help()

if __name__ == "__main__":
    main()