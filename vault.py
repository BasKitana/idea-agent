import re
import unicodedata
from datetime import date
from pathlib import Path

import yaml

RESERVED = {".obsidian", ".rag_index.json"}
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
WINDOWS_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x08\x0e-\x1f]')
MAX_SLUG_LENGTH = 100


def list_folders(vault_path: Path) -> list[str]:
    return sorted(
        p.name for p in vault_path.iterdir()
        if p.is_dir() and p.name not in RESERVED and not p.name.startswith(".")
    )


def slugify(text: str, default: str = "Untitled Idea") -> str:
    slug = WINDOWS_INVALID_CHARS.sub("", text)
    slug = "".join(ch for ch in slug if unicodedata.category(ch) != "Cf")
    slug = re.sub(r"[\s_]+", " ", slug).strip()
    slug = slug[:MAX_SLUG_LENGTH].rstrip(". ")
    if not slug:
        return default
    if slug.upper() in WINDOWS_RESERVED_NAMES:
        slug = f"Idea - {slug}"
    return slug


def unique_path(folder: Path, slug: str) -> Path:
    candidate = folder / f"{slug}.md"
    n = 2
    while candidate.exists():
        candidate = folder / f"{slug} ({n}).md"
        n += 1
    return candidate


def write_note(vault_path: Path, folder: str, title: str, tags: list[str],
               body: str, related_titles: list[str]) -> Path:
    # Route the LLM-provided folder through the same sanitizer as titles.
    # Critically, this strips every path separator (/ \ :), so `folder`
    # can never contain ".." or an absolute path and escape the vault --
    # it always collapses to a single safe segment.
    safe_folder = slugify(folder, default="Inbox")
    folder_path = vault_path / safe_folder
    folder_path.mkdir(parents=True, exist_ok=True)

    heading = title.replace("\n", " ").replace("\r", " ").strip() or "Untitled Idea"
    unique_tags = list(dict.fromkeys(tags)) if tags else []

    frontmatter = {
        "date": date.today().isoformat(),
        "type": "idea",
        "tags": unique_tags,
    }

    links = "".join(f"\n- [[{t}]]" for t in related_titles) if related_titles else "\n- none yet"

    content = (
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
        + "\n---\n\n"
        + f"# {heading}\n\n"
        + f"{body.strip()}\n\n"
        + "## Related\n"
        + links
        + "\n"
    )

    path = unique_path(folder_path, slugify(title))
    path.write_text(content, encoding="utf-8")
    return path
