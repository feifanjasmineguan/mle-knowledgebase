# src/embeddings/embedder.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from config import BASE_DIR

STORE_DIR = BASE_DIR / "vector_store"
VECTORIZER_PATH = STORE_DIR / "vectorizer.pkl"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks

def fit_and_embed(texts: list[str]) -> tuple[np.ndarray, TfidfVectorizer]:
    """Fit vectorizer on all chunks and return embeddings."""
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams
    )
    embeddings = vectorizer.fit_transform(texts).toarray()

    # Save vectorizer so we can embed queries later
    STORE_DIR.mkdir(exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    return embeddings

def embed_query(query: str) -> np.ndarray:
    """Embed a query using the saved vectorizer."""
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer.transform([query]).toarray()[0]