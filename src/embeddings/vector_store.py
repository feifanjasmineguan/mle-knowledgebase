# src/embeddings/vector_store.py
import numpy as np
import json
from pathlib import Path
from config import BASE_DIR

STORE_DIR = BASE_DIR / "vector_store"

def save_store(embeddings: np.ndarray, metadata: list[dict]):
    STORE_DIR.mkdir(exist_ok=True)
    np.save(STORE_DIR / "embeddings.npy", embeddings)
    with open(STORE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved {len(embeddings)} embeddings → vector_store/")

def load_store() -> tuple[np.ndarray, list[dict]]:
    embeddings = np.load(STORE_DIR / "embeddings.npy")
    with open(STORE_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return embeddings, metadata

def store_exists() -> bool:
    return (
        (STORE_DIR / "embeddings.npy").exists() and
        (STORE_DIR / "vectorizer.pkl").exists()
    )

def search(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    embeddings, metadata = load_store()

    # Cosine similarity — same math as before
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    scores = normalized @ query_norm

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {**metadata[i], "score": float(scores[i])}
        for i in top_indices
    ]