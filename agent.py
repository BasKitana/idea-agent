import json

import ollama

import config

SYSTEM_PROMPT = """You are an idea-filing assistant for a personal Obsidian vault.
Given a raw idea and the vault's current top-level folders, decide where it belongs.

Rules:
- Prefer an existing folder if the idea reasonably fits one.
- Only invent a new folder name if none of the existing folders fit at all.
- Folder names are short, Title Case, general categories (e.g. "Business", "Product", "Writing").
- Expand the raw idea into a few clear sentences of body text, but do not invent facts
  the user did not state.
- related_titles must only contain titles copied exactly from the provided related notes list,
  never invented. Leave it empty if none are truly relevant.

Respond with ONLY a JSON object with exactly these keys:
{"title": str, "folder": str, "tags": [str], "body": str, "related_titles": [str]}
"""


def _build_user_prompt(idea_text: str, existing_folders: list[str], related: list[dict]) -> str:
    related_lines = "\n".join(f"- {r['title']}" for r in related) or "(none found)"
    folder_lines = ", ".join(existing_folders) or "(vault is empty, no folders yet)"
    return (
        f"Existing folders: {folder_lines}\n\n"
        f"Related notes already in the vault:\n{related_lines}\n\n"
        f"Raw idea:\n{idea_text}"
    )


def _call_ollama(idea_text: str, existing_folders: list[str], related: list[dict]) -> dict:
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        format="json",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(idea_text, existing_folders, related)},
        ],
    )
    return json.loads(response["message"]["content"])


def classify_idea(idea_text: str, existing_folders: list[str], related: list[dict]) -> dict:
    for _ in range(2):
        try:
            result = _call_ollama(idea_text, existing_folders, related)
            return {
                "title": str(result["title"]).strip()[:120],
                "folder": str(result.get("folder") or "Inbox").strip(),
                "tags": [str(t) for t in result.get("tags", [])],
                "body": str(result.get("body") or idea_text).strip(),
                "related_titles": [str(t) for t in result.get("related_titles", [])],
            }
        except Exception:
            # Any failure here (bad JSON, non-dict response, connection drop) should
            # degrade to the Inbox fallback below, never crash the caller.
            continue

    return {
        "title": idea_text.strip()[:60] or "Untitled Idea",
        "folder": "Inbox",
        "tags": ["unsorted"],
        "body": idea_text.strip(),
        "related_titles": [],
    }
