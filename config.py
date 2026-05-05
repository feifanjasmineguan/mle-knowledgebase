# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Directories
RAW_DIR = BASE_DIR / "raw"
WIKI_DIR = BASE_DIR / "wiki"

# Topic
TOPIC = "Machine Learning Engineering"

# Models
COMPILE_MODEL = "claude-haiku-4-5-20251001"  # cheap, bulk summarization
QA_MODEL = "claude-sonnet-4-6"               # smarter, Q&A

# Token limits
COMPILE_MAX_TOKENS = 2048
QA_MAX_TOKENS = 4096

# Ingest settings
MAX_ARTICLE_CHARS = 32000  # truncate huge pages before sending to Claude