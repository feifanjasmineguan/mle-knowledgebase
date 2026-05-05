# src/compiler/summarizer.py
import anthropic
import re
from pathlib import Path
from dotenv import load_dotenv
from config import WIKI_DIR, TOPIC, COMPILE_MODEL, COMPILE_MAX_TOKENS

load_dotenv()
client = anthropic.Anthropic()

def summarize_source(source: dict) -> str:
    print(f"  Summarizing: {source['title']}...")

    message = client.messages.create(
        model=COMPILE_MODEL,
        max_tokens=COMPILE_MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": f"""You are building a personal knowledge base about "{TOPIC}".
Convert the following source into a well-structured Markdown wiki article.

Include these sections:
# [Title]
## Summary
## Key Concepts
## Details
## Interview Relevance (common questions this concept appears in)
## Related Topics (as [[WikiLinks]])

Keep it factual, dense, and focused on what an MLE interviewer would ask.
Source: {source['title']} ({source['source_type']})

Content:
{source['content']}"""
        }]
    )

    article = message.content[0].text

    # Save to wiki/
    WIKI_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r"[^a-z0-9]+", "_", source["title"].lower()).strip("_")
    out_path = WIKI_DIR / f"{safe_name}.md"
    out_path.write_text(article, encoding="utf-8")

    print(f"  ✓ Saved → wiki/{safe_name}.md")
    return str(out_path)