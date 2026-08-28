import re
import unicodedata
from datetime import date
from pathlib import Path

import yaml

import config

RESERVED = {".obsidian", ".rag_index.json"}
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
# Union of: Windows-forbidden filename chars, ASCII control chars (excluding
# tab/CR/LF, which get normalized to spaces instead of vanishing), and the
# characters Obsidian's own docs say can break wikilink parsing even where
# the OS itself would allow them: "# | ^ : %% [[ ]]".
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\[\]#^\x00-\x08\x0e-\x1f]')
MAX_SLUG_LENGTH = 100


class _IndentedDumper(yaml.SafeDumper):
    """PyYAML's default emits block-sequence list items flush-left with the
    parent key (`tags:\\n- a`), but Obsidian's own Properties UI writes them
    2-space indented (`tags:\\n  - a`). Both are valid YAML, but the mismatch
    means opening a generated note in Obsidian and touching its properties
    rewrites the indentation, producing a spurious diff. This is the standard
    PyYAML workaround to match that convention exactly."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def list_folders(vault_path: Path) -> list[str]:
    return sorted(
        p.name for p in vault_path.iterdir()
        if p.is_dir() and p.name not in RESERVED and not p.name.startswith(".")
    )


def slugify(text: str, default: str = "Untitled") -> str:
    slug = UNSAFE_FILENAME_CHARS.sub("", text)
    slug = "".join(ch for ch in slug if unicodedata.category(ch) != "Cf")
    slug = re.sub(r"[\s_]+", " ", slug).strip()
    slug = slug[:MAX_SLUG_LENGTH].rstrip(". ")
    if not slug:
        return default
    if slug.upper() in WINDOWS_RESERVED_NAMES:
        slug = f"Note - {slug}"
    return slug


def unique_path(folder: Path, slug: str) -> Path:
    candidate = folder / f"{slug}.md"
    n = 2
    while candidate.exists():
        candidate = folder / f"{slug} ({n}).md"
        n += 1
    return candidate


def _yaml_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _clean_title(title: str) -> str:
    return title.replace("\n", " ").replace("\r", " ").strip() or "Untitled"


def _wikilink(title: str) -> str:
    """A note's actual filename is slugify(title), not the raw title -- any
    title containing characters slugify strips/changes (or long enough to be
    truncated) would otherwise produce a [[link]] whose target text doesn't
    match the file it's meant to point at, silently failing to resolve.
    Target the real filename and keep the original as the alias/display text
    when they diverge (Obsidian's own recommended `[[target|display]]`
    pattern), so links still resolve correctly and still read naturally."""
    target = slugify(title)
    if target == title:
        return f"[[{target}]]"
    return f"[[{target}|{title}]]"


def _frontmatter(title: str, note_type: str, tags: list[str], extra: dict = None) -> str:
    unique_tags = list(dict.fromkeys(tags)) if tags else []
    lines = [
        f"title: {_yaml_quote(_clean_title(title))}",
        f"date: {date.today().isoformat()}",  # bare/unquoted -> native Obsidian Date property
        f"type: {note_type}",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"{key}: {_yaml_quote(str(value))}")
    tags_yaml = yaml.dump(
        {"tags": unique_tags}, sort_keys=False, allow_unicode=True, Dumper=_IndentedDumper,
    ).strip()
    lines.append(tags_yaml)
    return "---\n" + "\n".join(lines) + "\n---\n"


def _links_section(heading: str, links: list[str]) -> str:
    body = "".join(f"\n- {_wikilink(t)}" for t in links) if links else "\n- none yet"
    return f"## {heading}\n{body}\n"


def write_note(vault_path: Path, note_type: str, title: str, tags: list[str],
                body: str, links: list[str], related_heading: str = "Related") -> Path:
    folder_name = config.FOLDER_BY_TYPE.get(note_type, config.FOLDER_BY_TYPE["log"])
    folder_path = vault_path / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    heading = _clean_title(title)
    content = (
        _frontmatter(title, note_type, tags)
        + f"\n# {heading}\n\n"
        + f"{body.strip()}\n\n"
        + _links_section(related_heading, links)
    )

    path = unique_path(folder_path, slugify(title))
    path.write_text(content, encoding="utf-8")
    return path


def append_daily_log_links(vault_path: Path, new_note_titles: list[str],
                            duplicate_notes: list[tuple[str, str]]) -> Path:
    """Ensure today's log note exists and links to everything captured today.

    This is the one deliberately bidirectional link in the system: the log
    is a purpose-built index of its own day, not a peer content note, so a
    new note also linking back to the log (written separately, in the note's
    own `## Related` section) is intentional rather than redundant -- it's
    what guarantees no created note is a true orphan with zero inbound links.

    `duplicate_notes` is a list of (raw_capture, existing_title) pairs: a
    capture recognized as already covered gets logged here with a link to
    the existing note, rather than ever editing that existing note.
    """
    folder_path = vault_path / config.FOLDER_BY_TYPE["log"]
    folder_path.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = folder_path / f"{today}.md"

    if not path.exists():
        content = _frontmatter(today, "log", ["daily"]) + f"\n# {today}\n\n## Captured today\n"
        path.write_text(content, encoding="utf-8")

    text = path.read_text(encoding="utf-8")
    additions = "".join(f"\n- {_wikilink(t)}" for t in new_note_titles)
    additions += "".join(
        f"\n- Already covered: {raw[:80]!r} -> {_wikilink(existing)}"
        for raw, existing in duplicate_notes
    )
    if additions:
        text = text.rstrip("\n") + "\n" + additions.strip("\n") + "\n"
        path.write_text(text, encoding="utf-8")

    return path
