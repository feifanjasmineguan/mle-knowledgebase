# src/qa/query_handler.py
import anthropic
from datetime import datetime
from dotenv import load_dotenv
from config import WIKI_DIR, TOPIC, QA_MODEL, QA_MAX_TOKENS
from src.embeddings.embedder import embed_query
from src.embeddings.vector_store import search, store_exists

load_dotenv()
client = anthropic.Anthropic()

def ask_wiki(question: str) -> str:
    if not store_exists():
        return "Vector store not found. Run 'python main.py build:store' first."

    # Embed question and retrieve relevant chunks only
    query_vec = embed_query(question)
    results = search(query_vec, top_k=5)

    sources_used = list(set(r["source"] for r in results))
    print(f"Found relevant chunks from: {sources_used}")

    context = "\n\n---\n\n".join(
        f"[{r['source']}] (score: {r['score']:.2f})\n{r['chunk']}"
        for r in results
    )

    message = client.messages.create(
        model=QA_MODEL,
        max_tokens=QA_MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": f"""You are an expert ML interview coach.
Answer the question using only the knowledge base excerpts below.
Cite sources as [article_name]. Include likely follow-up interview questions.
Format your answer in clear Markdown.

Knowledge base excerpts:
{context}

Question: {question}"""
        }]
    )

    answer = message.content[0].text

    # File the answer back into wiki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qa_path = WIKI_DIR / f"_qa_{timestamp}.md"
    qa_path.write_text(f"# Q: {question}\n\n{answer}", encoding="utf-8")

    return answer