# src/compiler/linker.py
import re
from collections import defaultdict
from config import WIKI_DIR

def add_backlinks():
    md_files = list(WIKI_DIR.glob("*.md"))
    link_map = defaultdict(list)  # target -> [files that link to it]

    # Find all [[links]] across all files
    for f in md_files:
        content = f.read_text(encoding="utf-8")
        links = re.findall(r"\[\[([^\]]+)\]\]", content)
        for link in links:
            link_map[link.lower()].append(f.stem)

    # Append backlinks section to each target file
    for f in md_files:
        key = f.stem.lower()
        sources = link_map.get(key, [])
        if not sources:
            continue

        content = f.read_text(encoding="utf-8")
        if "## Backlinks" in content:
            continue  # already has backlinks

        backlinks = "\n".join(f"- [[{s}]]" for s in sources)
        content += f"\n\n## Backlinks\n{backlinks}"
        f.write_text(content, encoding="utf-8")

    print(f"✓ Backlinks added to {len(link_map)} articles")