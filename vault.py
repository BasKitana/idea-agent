import re
from datetime import date
from pathlib import Path

import yaml

RESERVED = {".obsidian", ".rag_index.json"}
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def list_folders(vault_path: Path) -> list[str]:
    return sorted(
        p.name for p in vault_path.iterdir()
        if p.is_dir() and p.name not in RESERVED and not p.name.startswith(".")
    )


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title).strip()
    slug = re.sub(r"[\s_]+", " ", slug).strip()
    slug = slug or "Untitled Idea"
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
    folder_path = vault_path / folder
    folder_path.mkdir(parents=True, exist_ok=True)

    frontmatter = {
        "date": date.today().isoformat(),
        "type": "idea",
        "tags": tags or [],
    }

    links = "".join(f"\n- [[{t}]]" for t in related_titles) if related_titles else "\n- none yet"

    content = (
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False).strip()
        + "\n---\n\n"
        + f"# {title}\n\n"
        + f"{body.strip()}\n\n"
        + "## Related\n"
        + links
        + "\n"
    )

    path = unique_path(folder_path, slugify(title))
    path.write_text(content, encoding="utf-8")
    return path
