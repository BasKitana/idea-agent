import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_vault_path = os.environ.get("VAULT_PATH")
if not _vault_path:
    raise SystemExit(
        "VAULT_PATH is not set. Copy .env.example to .env and point it at your Obsidian vault."
    )
VAULT_PATH = Path(_vault_path)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "5"))
INDEX_PATH = VAULT_PATH / ".rag_index.json"
