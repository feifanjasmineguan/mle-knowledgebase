# src/compiler/index_builder.py
import anthropic
from dotenv import load_dotenv
from config import WIKI_DIR, TOPIC, COMPILE_MODEL

load_dotenv()
client = anthropic.Anthropic()

def build_index():
    md_files = [f for f in WIKI_DIR.glob("*.md") if f.name != "_index.md"]

    if not md_files:
        print("No wiki articles found. Ingest some sources first.")
        return

    print(f"Building index from {len(md_files)} articles...")

    # Grab first 500 chars of each article as a preview
    previews = []
    for f in md_files:
        content = f.read_text(encoding="utf-8")
        previews.append(f"### {f.stem}\n{content[:500]}...")

    message = client.messages.create(
        model=COMPILE_MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Create a master index for a knowledge base about "{TOPIC}".

Group these articles into logical categories (e.g. Classical ML, Deep Learning,
Training & Optimization, Systems, Statistics). For each article write a one-line
description and link it as [[article_name]].

Articles:
{"".join(previews)}"""
        }]
    )

    index_path = WIKI_DIR / "_index.md"
    index_path.write_text(message.content[0].text, encoding="utf-8")
    print(f"✓ Index built → wiki/_index.md")