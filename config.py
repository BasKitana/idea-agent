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
INDEX_PATH = VAULT_PATH / ".rag_index.json"

# Closed set of note types -> the exact folder each is routed to. Folder is
# never free text from the LLM (old design let it invent folder names, which
# both biased everything toward "Inbox" and was a path-traversal surface).
FOLDER_BY_TYPE = {
    "concept": "01_Concepts",
    "project": "02_Projects",
    "entity": "03_Entities",
    "log": "04_Logs",
}

# Calibrated empirically against all-MiniLM-L6-v2 (see scratchpad/calibrate.py):
# a real paraphrase scored as low as 0.552, a genuinely different-but-related
# idea scored as high as 0.516 -- the bands overlap, so no single cosine cutoff
# can decide "same idea" on its own. DEDUP_RETRIEVE_SCORE is deliberately a low
# recall-oriented bar for pulling candidates; the LLM then judges each one.
DEDUP_RETRIEVE_SCORE = float(os.environ.get("DEDUP_RETRIEVE_SCORE", "0.40"))
DEDUP_TOP_K = int(os.environ.get("DEDUP_TOP_K", "8"))

# Below this word count, forced to exactly one note -- the local model was
# observed inventing 4-5 fictional sub-topics when asked to atomize vague
# one-line input instead of just capturing it as-is.
ATOMIZE_MIN_WORDS = int(os.environ.get("ATOMIZE_MIN_WORDS", "12"))

# A candidate scoring at or above this is auto-linked regardless of whether
# the LLM's own "links" field mentioned it. Measured directly: given a
# clearly-related candidate right there in the prompt, the model only
# populated "links" in 4/6 runs -- it forgets, the same unreliability pattern
# as the earlier duplicate-detection bug. Since the retrieval score is known
# before the LLM even runs, don't depend on it remembering; this is a floor
# under the LLM's own judgment, not a replacement for it. 0.50 sits below the
# real related-note case measured (0.585) with margin above the noise band
# (0.038-0.456) from earlier calibration.
AUTO_LINK_SCORE = float(os.environ.get("AUTO_LINK_SCORE", "0.50"))
