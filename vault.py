import re
import unicodedata
from datetime import date
from pathlib import Path

import send2trash
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


def find_existing_note(vault_path: Path, title: str) -> Path | None:
    """The note already occupying this title's filename, anywhere in the
    vault, or None.

    Deliberately vault-wide rather than per-folder: Obsidian resolves
    [[wikilinks]] by filename across the entire vault, so two "X.md" files in
    different type folders aren't two notes -- they're one ambiguous link
    target, and which one a link opens is not something the writer controls.
    A title collision across folders is therefore just as broken as one
    within a folder."""
    filename = f"{slugify(title)}.md"
    for folder in config.FOLDER_BY_TYPE.values():
        candidate = vault_path / folder / filename
        if candidate.exists():
            return candidate
    return None


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
    """Create the note, or merge into the one already holding that filename.

    Never produces "X (2).md". An earlier version disambiguated collisions
    with a numeric suffix, which is what produced the reported
    "Idea Agent Project" / "Idea Agent Project (2)" pair: two files with the
    same title are not two notes, they're one subject split in half, and the
    suffixed twin is unreachable by [[wikilink]] anyway since Obsidian
    resolves those by filename. Merging is both the safe outcome (nothing is
    overwritten -- append_update only inserts) and the one the rest of this
    tool is built around."""
    existing = find_existing_note(vault_path, title)
    if existing:
        return append_update(vault_path, existing.relative_to(vault_path).as_posix(), body)

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

    path = folder_path / f"{slugify(title)}.md"
    path.write_text(content, encoding="utf-8")
    return path


def append_update(vault_path: Path, rel_path: str, text: str) -> Path:
    """Add a new fact to an existing note without touching anything already
    there: read the full current content, insert one new bullet under an
    "## Updates" heading (added once, at the end, if it doesn't exist yet),
    write the full content back. Never rewrites, reorders, or removes any
    existing line -- the one operation in this codebase that touches an
    existing note at all, so it's deliberately this narrow.

    Raises FileNotFoundError if the target no longer exists (e.g. deleted
    between candidate retrieval and this call) -- callers should treat that
    like any other per-item failure, not attempt to recover it here.
    """
    path = vault_path / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{rel_path} no longer exists")

    existing = path.read_text(encoding="utf-8")
    today = date.today().isoformat()
    bullet = f"- ({today}) {text.strip()}"

    marker = "## Updates\n"
    idx = existing.find(marker)
    if idx == -1:
        new_content = existing.rstrip("\n") + "\n\n" + marker + bullet + "\n"
    else:
        insert_at = idx + len(marker)
        new_content = existing[:insert_at] + bullet + "\n" + existing[insert_at:]

    path.write_text(new_content, encoding="utf-8")
    return path


_SECTION_RE = re.compile(r"^## ", re.MULTILINE)


def append_to_body(vault_path: Path, rel_path: str, text: str) -> Path:
    """Add a sentence to the note's main body -- the prose under the H1,
    before the first "## " section -- rather than as a dated Updates bullet.

    For content that belongs to what the note *is* rather than to what
    happened to it since. Still purely additive: nothing already written is
    rewritten or removed."""
    path = vault_path / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{rel_path} no longer exists")

    existing = path.read_text(encoding="utf-8")
    addition = text.strip()
    if not addition:
        return path

    match = _SECTION_RE.search(existing)
    if match:
        head, tail = existing[:match.start()], existing[match.start():]
        new_content = head.rstrip("\n") + f"\n{addition}\n\n" + tail
    else:
        new_content = existing.rstrip("\n") + f"\n{addition}\n"

    path.write_text(new_content, encoding="utf-8")
    return path


def rename_note(vault_path: Path, rel_path: str, new_title: str) -> Path:
    """Retitle a note and repoint every [[wikilink]] in the vault at it.

    Returns the note's path -- the new one, or the unchanged original when
    the rename is refused. Refused rather than forced when the new filename
    is already taken, since silently merging two notes is not what a retitle
    asked for, and the no-duplicate-filenames invariant must hold either way.

    Rewriting inbound links is not optional: Obsidian resolves links by
    filename, so a rename without it would leave every existing reference
    pointing at a file that no longer exists."""
    path = vault_path / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{rel_path} no longer exists")

    old_slug = path.stem
    new_slug = slugify(new_title)
    if new_slug == old_slug:
        return path
    taken = find_existing_note(vault_path, new_title)
    if taken and taken.resolve() != path.resolve():
        return path

    content = path.read_text(encoding="utf-8")
    # [ \t]* rather than \s* throughout: \s matches newlines, so a greedy
    # \s*$ swallows the blank line after the heading and glues the body onto
    # it (caught live -- the H1 ended up directly above the first paragraph).
    content = re.sub(
        r"^title:[ \t]*.*$", f"title: {_yaml_quote(_clean_title(new_title))}",
        content, count=1, flags=re.MULTILINE,
    )
    content = re.sub(
        rf"^#[ \t]+{re.escape(old_slug)}[ \t]*$", f"# {_clean_title(new_title)}",
        content, count=1, flags=re.MULTILINE,
    )
    new_path = path.with_name(f"{new_slug}.md")
    new_path.write_text(content, encoding="utf-8")
    path.unlink()

    link_re = re.compile(rf"\[\[{re.escape(old_slug)}(\|[^\]]*)?\]\]")
    replacement = _wikilink(new_title)
    for folder in config.FOLDER_BY_TYPE.values():
        folder_path = vault_path / folder
        if not folder_path.is_dir():
            continue
        for md in folder_path.glob("*.md"):
            text = md.read_text(encoding="utf-8")
            updated = link_re.sub(replacement.replace("\\", "\\\\"), text)
            if updated != text:
                md.write_text(updated, encoding="utf-8")

    return new_path


def add_link(vault_path: Path, rel_path: str, target_title: str, heading: str = "Related") -> Path:
    """Additively insert a wikilink to target_title into an existing note's
    section (creating the heading once if it isn't there yet). Idempotent --
    a no-op if the link is already present. Never rewrites, reorders, or
    removes anything already there."""
    path = vault_path / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{rel_path} no longer exists")

    existing = path.read_text(encoding="utf-8")
    link = _wikilink(target_title)
    if link in existing or f"[[{slugify(target_title)}]]" in existing:
        return path

    line = f"- {link}"
    marker = f"## {heading}\n"
    idx = existing.find(marker)
    if idx == -1:
        new_content = existing.rstrip("\n") + "\n\n" + marker + line + "\n"
    else:
        insert_at = idx + len(marker)
        new_content = existing[:insert_at] + line + "\n" + existing[insert_at:]

    path.write_text(new_content, encoding="utf-8")
    return path


def delete_note(vault_path: Path, rel_path: str) -> None:
    """Send an existing note to the OS recycle bin -- never a permanent,
    unrecoverable delete. Raises FileNotFoundError if it's already gone."""
    path = vault_path / rel_path
    if not path.exists():
        raise FileNotFoundError(f"{rel_path} no longer exists")
    send2trash.send2trash(str(path))


def append_daily_log_links(vault_path: Path, new_note_titles: list[str],
                            duplicate_notes: list[tuple[str, str]],
                            updated_notes: list[tuple[str, str]] = None) -> Path:
    """Ensure today's log note exists and links to everything captured today.

    This is the one deliberately bidirectional link in the system: the log
    is a purpose-built index of its own day, not a peer content note, so a
    new note also linking back to the log (written separately, in the note's
    own `## Related` section) is intentional rather than redundant -- it's
    what guarantees no created note is a true orphan with zero inbound links.

    `duplicate_notes` is a list of (raw_capture, existing_title) pairs: a
    capture recognized as already covered gets logged here with a link to
    the existing note, rather than ever editing that existing note.

    `updated_notes` is a list of (raw_capture, target_title) pairs: a
    capture merged into an existing note via append_update() gets logged
    here too, so today's log stays a complete record of what happened even
    though no new file was created for it.
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
    additions += "".join(
        f"\n- Updated {_wikilink(target)}: {raw[:80]!r}"
        for raw, target in (updated_notes or [])
    )
    if additions:
        text = text.rstrip("\n") + "\n" + additions.strip("\n") + "\n"
        path.write_text(text, encoding="utf-8")

    return path
