# src/embeddings/build_store.py
from config import WIKI_DIR
from src.embeddings.embedder import chunk_text, fit_and_embed
from src.embeddings.vector_store import save_store

def build_vector_store():
    md_files = [
        f for f in WIKI_DIR.glob("*.md")
        if not f.name.startswith("_qa_")
    ]

    if not md_files:
        print("No wiki articles found. Run compile first.")
        return

    print(f"Building vector store from {len(md_files)} articles...")

    all_chunks = []
    all_metadata = []

    for f in md_files:
        content = f.read_text(encoding="utf-8")
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": f.stem,
                "chunk_index": i,
                "chunk": chunk,
            })

    print(f"Fitting TF-IDF on {len(all_chunks)} chunks...")
    embeddings = fit_and_embed(all_chunks)
    save_store(embeddings, all_metadata)
    print("✓ Vector store ready")