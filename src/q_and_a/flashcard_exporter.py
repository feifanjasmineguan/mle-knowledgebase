# src/qa/flashcard_exporter.py
import anthropic
import csv
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from config import WIKI_DIR, TOPIC, QA_MODEL, BASE_DIR

load_dotenv()
client = anthropic.Anthropic()

CARDS_DIR = BASE_DIR / "flashcards"

def generate_cards_for_article(article_path: Path) -> list[dict]:
    """Ask Claude to generate flashcards from one wiki article."""
    content = article_path.read_text(encoding="utf-8")

    message = client.messages.create(
        model=QA_MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""You are creating Anki flashcards for ML interview prep from this wiki article.

Generate 5-8 flashcards covering the most important concepts.
Focus on definitions, intuitions, and things interviewers commonly ask.

Return ONLY valid JSON in this exact format, no other text:
{{
  "cards": [
    {{
      "front": "What is the vanishing gradient problem?",
      "back": "During backpropagation gradients shrink exponentially through layers...",
      "tags": ["deep-learning", "training", "gradients"]
    }}
  ]
}}

Article ({article_path.stem}):
{content[:6000]}"""
        }]
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if Claude added them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw.strip())
    cards = data.get("cards", [])

    # Tag each card with its source article
    for card in cards:
        card["source"] = article_path.stem

    return cards


def export_flashcards(output_format: str = "csv"):
    """Generate flashcards from all wiki articles and export."""
    md_files = [
        f for f in WIKI_DIR.glob("*.md")
        if not f.name.startswith("_")  # skip _index.md and _qa_ files
    ]

    if not md_files:
        print("No wiki articles found. Run compile first.")
        return

    CARDS_DIR.mkdir(exist_ok=True)
    all_cards = []

    for i, f in enumerate(md_files, 1):
        print(f"[{i}/{len(md_files)}] Generating cards for: {f.stem}")
        try:
            cards = generate_cards_for_article(f)
            all_cards.extend(cards)
            print(f"  ✓ {len(cards)} cards generated")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    if not all_cards:
        print("No cards generated.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_format == "csv":
        # Anki imports CSV as: front, back, tags
        out_path = CARDS_DIR / f"flashcards_{timestamp}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Front", "Back", "Tags", "Source"])
            for card in all_cards:
                tags = " ".join(card.get("tags", []))
                writer.writerow([
                    card["front"],
                    card["back"],
                    tags,
                    card.get("source", ""),
                ])
        print(f"\n✓ {len(all_cards)} cards exported → {out_path}")
        print("Import into Anki: File → Import → select the CSV")

    elif output_format == "json":
        out_path = CARDS_DIR / f"flashcards_{timestamp}.json"
        out_path.write_text(json.dumps(all_cards, indent=2), encoding="utf-8")
        print(f"\n✓ {len(all_cards)} cards exported → {out_path}")

    elif output_format == "md":
        # Markdown format — readable in VSCode
        out_path = CARDS_DIR / f"flashcards_{timestamp}.md"
        lines = [f"# Flashcards — {TOPIC}\n", f"Generated: {timestamp}\n\n---\n"]
        for i, card in enumerate(all_cards, 1):
            lines.append(f"## Card {i} — {card.get('source', '')}")
            lines.append(f"\n**Q:** {card['front']}\n")
            lines.append(f"**A:** {card['back']}\n")
            tags = " ".join(f"`{t}`" for t in card.get("tags", []))
            lines.append(f"Tags: {tags}\n\n---\n")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✓ {len(all_cards)} cards exported → {out_path}")

    return str(out_path)