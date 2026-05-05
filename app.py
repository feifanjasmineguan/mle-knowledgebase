# app.py
import streamlit as st
from pathlib import Path
from config import WIKI_DIR, TOPIC
from src.q_and_a.query_handler import ask_wiki
from src.ingest.web_clipper import clip_url
from src.compiler.summarizer import summarize_source
from src.compiler.index_builder import build_index
from src.compiler.linker import add_backlinks
from src.embeddings.build_store import build_vector_store
from src.embeddings.vector_store import store_exists

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Knowledge Base",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 ML Interview Knowledge Base")
st.caption(f"Topic: {TOPIC}")

# ── Sidebar — status + ingest ───────────────────────────────────
with st.sidebar:
    st.header("Knowledge Base Status")

    md_files = [
        f for f in WIKI_DIR.glob("*.md")
        if not f.name.startswith("_")
    ] if WIKI_DIR.exists() else []

    st.metric("Wiki articles", len(md_files))
    st.metric("Vector store", "✓ Ready" if store_exists() else "✗ Not built")

    st.divider()

    # ── Ingest a URL ───────────────────────────────────────────
    st.subheader("Add a source")
    new_url = st.text_input("Paste a URL to ingest")

    if st.button("Ingest URL", disabled=not new_url):
        with st.spinner("Fetching and summarizing..."):
            try:
                source = clip_url(new_url)
                summarize_source(source)
                build_index()
                add_backlinks()
                build_vector_store()
                st.success(f"✓ Ingested: {source['title']}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # ── Rebuild controls ───────────────────────────────────────
    st.subheader("Maintenance")

    if st.button("Rebuild index + backlinks"):
        with st.spinner("Compiling..."):
            build_index()
            add_backlinks()
        st.success("✓ Index rebuilt")

    if st.button("Rebuild vector store"):
        with st.spinner("Embedding articles..."):
            build_vector_store()
        st.success("✓ Vector store rebuilt")

    st.divider()

    # ── Wiki article list ──────────────────────────────────────
    st.subheader("Wiki articles")
    if md_files:
        for f in sorted(md_files):
            st.markdown(f"- `{f.stem}`")
    else:
        st.info("No articles yet. Ingest a URL to get started.")

# ── Main panel — tabs ───────────────────────────────────────────
tab_qa, tab_wiki, tab_index = st.tabs(["💬 Ask", "📄 Browse Wiki", "🗂 Index"])

# ── Tab 1: Q&A ─────────────────────────────────────────────────
with tab_qa:
    st.subheader("Ask an interview question")

    # Suggested starter questions
    suggestions = [
        "What is the vanishing gradient problem and how do LSTMs solve it?",
        "Explain the bias-variance tradeoff",
        "Walk me through backpropagation step by step",
        "What is attention and why does it matter?",
        "Compare gradient descent optimizers: SGD, Adam, RMSProp",
    ]

    st.markdown("**Suggested questions:**")
    cols = st.columns(len(suggestions))
    for col, q in zip(cols, suggestions):
        if col.button(q[:40] + "...", help=q):
            st.session_state["question"] = q

    question = st.text_area(
        "Your question",
        value=st.session_state.get("question", ""),
        height=80,
        placeholder="e.g. Explain the transformer architecture and its key components",
    )

    if st.button("Ask", type="primary", disabled=not question):
        if not store_exists():
            st.warning("Vector store not built yet. Click 'Rebuild vector store' in the sidebar first.")
        else:
            with st.spinner("Searching wiki and generating answer..."):
                answer = ask_wiki(question)
            st.markdown("---")
            st.markdown(answer)

            # Show which articles were used
            st.caption("Answer saved to wiki/_qa_*.md")

    # ── Q&A history ───────────────────────────────────────────
    qa_files = sorted(WIKI_DIR.glob("_qa_*.md"), reverse=True) if WIKI_DIR.exists() else []
    if qa_files:
        with st.expander(f"Previous Q&A sessions ({len(qa_files)})"):
            selected = st.selectbox(
                "Select a session",
                qa_files,
                format_func=lambda f: f.stem,
            )
            if selected:
                st.markdown(selected.read_text(encoding="utf-8"))

# ── Tab 2: Browse wiki articles ────────────────────────────────
with tab_wiki:
    st.subheader("Browse wiki articles")

    if not md_files:
        st.info("No articles yet.")
    else:
        selected_article = st.selectbox(
            "Select an article",
            md_files,
            format_func=lambda f: f.stem.replace("_", " ").title(),
        )
        if selected_article:
            content = selected_article.read_text(encoding="utf-8")
            st.markdown(content)

# ── Tab 3: Master index ────────────────────────────────────────
with tab_index:
    st.subheader("Master index")
    index_path = WIKI_DIR / "_index.md"
    if index_path.exists():
        st.markdown(index_path.read_text(encoding="utf-8"))
    else:
        st.info("No index yet. Click 'Rebuild index + backlinks' in the sidebar.")