# src/ingest/web_clipper.py
import requests
from bs4 import BeautifulSoup
from config import MAX_ARTICLE_CHARS

def clip_url(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15) # fetch the webpage
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")  # parse url

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url.split("/")[-1]
    content = soup.get_text(separator="\n", strip=True) # extracts pain text

    # returns the same format as pdf_loader and note_importer
    return {
        "title": title,
        "content": content[:MAX_ARTICLE_CHARS],
        "source_type": "web",
        "source_url": url,
    }