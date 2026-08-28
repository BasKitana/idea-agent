import json

import ollama

import config

VALID_TYPES = set(config.FOLDER_BY_TYPE.keys())

# Split into two focused calls rather than one combined schema. Measured
# directly: a single call asking the model to both judge duplication AND
# generate atomic notes in one JSON response strongly biased it toward the
# generative "create" branch -- it missed an exact-match duplicate in every
# run, on both a 7B and a 14B model, so this is a structural prompt-design
# issue, not a model-capacity one. A plain yes/no judgment against retrieved
# candidates is a much more reliable task for a local model than a rare
# branch inside a richer generation schema.

META_COMMAND_SYSTEM_PROMPT = """You judge whether input to a note-filing tool is genuine
content worth recording, or an instruction directed at the TOOL ITSELF -- telling it to
create/edit/delete/organize/manage its own notes, files, or folders.

The critical distinction: is this about THIS TOOL'S OWN VAULT/FILES (an instruction), or about
literally anything else the user wants to build, do, learn, or remember in the real world
(content)? A capture describing a software project, app, or system the user wants to build is
ALWAYS content, even when phrased with "build"/"make"/"create" -- those verbs alone do NOT
mean it's an instruction. Only classify as an instruction when the ACTION is explicitly aimed
at the tool's own notes/files/folders/vault, not at some other project or system.

Examples of instructions (is_instruction: true) -- all explicitly about the tool's own notes/files:
"Make me a file that links to another file." "Edit that note and add X." "Delete the old
notes." "Create an empty folder [in the vault]." "Organize my vault." "Make a project folder
with no notes inside [the vault]."

Examples of genuine content (is_instruction: false) -- these describe projects/ideas to build
or remember, NOT the tool's own files, even though they use "build"/"make"/"create":
"I want to build a RAG pipeline using ChromaDB." "Build an app that tracks water intake."
"A research project where I collect data from a GitHub repository." "Create a marketing plan
for the launch." "Vercel's edge functions look good for the auth layer." "Remember to follow
up with Sarah about Clerk."

Respond with ONLY a JSON object:
{"is_instruction": true or false}
"""

DUPLICATE_SYSTEM_PROMPT = """You judge whether a new capture is the SAME IDEA as one of a list
of existing notes. This is a strict equivalence check, not a topic/relevance check.

Answer yes only if the capture and a candidate are clearly describing the same underlying
idea, just possibly worded differently. Two different ideas about the same general subject
are NOT the same idea -- answer no for those, even if closely related.

Respond with ONLY a JSON object:
{"duplicate_of": str or null}
If yes, duplicate_of is the matching candidate's EXACT title, copied verbatim. If no single
candidate is genuinely the same idea, duplicate_of is null.
"""

ATOMIZE_SYSTEM_PROMPT = """You decompose a raw capture into atomic notes for a technical
Obsidian vault. The capture has already been confirmed to NOT duplicate any existing note.

NOTE TYPES (choose exactly one per note):
- concept: a broad technical principle, algorithm, or academic idea
- project: an active development task, roadmap, or system specification
- entity: a specific API, organization, person, or distinct component
- log: a chronological record of something that happened (a meeting, an event)

ATOMICITY:
- If the capture describes ONE distinct concept, produce exactly ONE note. Do not invent
  additional notes that aren't actually in the input.
- If the capture genuinely contains multiple distinct concepts (e.g. a project plus a specific
  technology choice plus something to learn), split into multiple atomic notes, one per
  concept, and link them to each other via "links".
- If the capture is vague, short, or a placeholder with no real distinguishable topic, produce
  exactly ONE note that captures it as-is. NEVER fabricate sub-topics, structure, or
  elaboration not actually present in the input. Inventing content the user didn't say is a
  serious failure, worse than under-splitting.

You may also be given RELATED notes (not duplicates, just related) -- link to them by exact
title when a note you create is genuinely, specifically connected to one.

WRITING STYLE:
- Information-dense, concise, technical. No fluff, no filler sentences.
- title: specific and descriptive, not a broad category name.
- tags: 1-4 short lowercase-hyphenated tags.
- body: 1-4 sentences. State only what the capture actually said; do not add facts, numbers,
  or specifics the user did not provide.
- links: titles of OTHER notes (related notes given to you, or sibling notes in this same
  response) this note is genuinely, specifically related to. Copy titles exactly, never
  invent one. Leave empty if nothing is genuinely related.

Respond with ONLY a JSON object:
{"notes": [{"title": str, "type": "concept|project|entity|log", "tags": [str],
"body": str, "links": [str]}]}
"""


def _format_candidates(candidates: list[dict]) -> str:
    if not candidates:
        return "(none found)"
    return "\n".join(
        f'- "{c["title"]}" (type: {c.get("type", "concept")}): {c.get("excerpt", "")}'
        for c in candidates
    )


def _vague_guard(capture_text: str) -> str:
    if len(capture_text.split()) < config.ATOMIZE_MIN_WORDS:
        return ("\nThis capture is short/vague. Produce exactly ONE note. Do not split it, "
                  "do not invent sub-topics.\n")
    return ""


def _call_meta_check(capture_text: str) -> dict:
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": META_COMMAND_SYSTEM_PROMPT},
            {"role": "user", "content": capture_text},
        ],
    )
    return json.loads(response["message"]["content"])


def _call_duplicate_check(capture_text: str, candidates: list[dict]) -> dict:
    prompt = f"Existing notes:\n{_format_candidates(candidates)}\n\nNew capture:\n{capture_text}"
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": DUPLICATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _call_atomize(capture_text: str, candidates: list[dict]) -> dict:
    prompt = (
        f"Related notes (not duplicates):\n{_format_candidates(candidates)}\n"
        f"{_vague_guard(capture_text)}\nRaw capture:\n{capture_text}"
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": ATOMIZE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _as_str(value, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return []


def _normalize_note(item: dict, fallback_title: str, linkable_titles: set) -> dict:
    item = {str(k).lower(): v for k, v in item.items()}
    links = _as_str_list(item.get("links"))
    note_type = _as_str(item.get("type"), "concept").lower()
    if note_type not in VALID_TYPES:
        note_type = "concept"
    return {
        "action": "create",
        "title": _as_str(item.get("title"), fallback_title)[:120],
        "type": note_type,
        "tags": _as_str_list(item.get("tags")),
        "body": _as_str(item.get("body"), fallback_title),
        "links": [t for t in links if t in linkable_titles],
    }


def _is_meta_command(capture_text: str) -> bool:
    """True if the capture is an instruction directed at the tool (create,
    edit, delete, organize) rather than genuine content to record. Fails
    open toward "not an instruction" -- an LLM error should never silently
    swallow real content the user typed."""
    try:
        result = _call_meta_check(capture_text)
        if not isinstance(result, dict):
            return False
        return bool(result.get("is_instruction", False))
    except Exception:
        return False


def _one_duplicate_vote(capture_text: str, candidates: list[dict], candidate_titles: set) -> str | None:
    try:
        result = _call_duplicate_check(capture_text, candidates)
        if not isinstance(result, dict):
            return None
        target = _as_str(result.get("duplicate_of"), "")
        return target if target in candidate_titles else None
    except Exception:
        return None


def _check_duplicate(capture_text: str, candidates: list[dict], votes: int = 3) -> str | None:
    """Returns the exact title of a genuine duplicate candidate, or None.

    Measured directly: a single call to the local model missed an exact-match
    duplicate about half the time, and a 2-of-3 MAJORITY vote made this worse
    (1/8 caught, not better), which means the model's true per-call "yes"
    rate is well under 50% -- a systematic bias toward "no", not noise around
    a coin flip. Majority voting compounds an under-triggering bias; the
    correct fix is an OR-ensemble (any positive vote counts), which gives the
    "yes" branch multiple independent chances to fire. Confirmed empirically
    before shipping (see scratchpad/dup_reliability.py results). Both
    outcomes are non-destructive by design -- a duplicate never edits or
    deletes anything, it only logs and links -- so biasing toward catching
    duplicates over precision is the right tradeoff here.
    """
    if not candidates:
        return None
    candidate_titles = {c["title"] for c in candidates}
    for _ in range(votes):
        vote = _one_duplicate_vote(capture_text, candidates, candidate_titles)
        if vote:
            return vote
    return None


def _atomize(capture_text: str, candidates: list[dict]) -> list[dict]:
    candidate_titles = {c["title"] for c in candidates}
    fallback_title = capture_text.strip()[:60] or "Untitled"

    for _ in range(2):
        try:
            result = _call_atomize(capture_text, candidates)
            if not isinstance(result, dict):
                continue
            result = {str(k).lower(): v for k, v in result.items()}
            raw_notes = result.get("notes")
            if not isinstance(raw_notes, list) or not raw_notes:
                continue

            # Per-note fallback titles must be unique even when several sibling
            # notes are all missing a title: two notes sharing a fallback would
            # collide to the same filename (unique_path() disambiguates that
            # fine), but Obsidian resolves [[links]] by filename, not display
            # text, so a link meant for the second note would silently resolve
            # to the first instead. Index-suffix each fallback when needed.
            fallbacks = [
                fallback_title if len(raw_notes) == 1 else f"{fallback_title} ({i + 1})"
                for i in range(len(raw_notes))
            ]
            sibling_titles = {
                _as_str(n.get("title"), fb)[:120] for n, fb in zip(raw_notes, fallbacks)
            }
            linkable = candidate_titles | sibling_titles
            notes = [
                _normalize_note(n, fb, linkable) for n, fb in zip(raw_notes, fallbacks)
            ]
            if notes:
                return notes
        except Exception:
            continue

    return [{
        "action": "create", "title": fallback_title, "type": "concept",
        "tags": ["unsorted"], "body": capture_text.strip(), "links": [],
    }]


def process_capture(capture_text: str, candidates: list[dict]) -> list[dict]:
    """Returns a list of items, each one of:
    {"action": "create", "title", "type", "tags", "body", "links"},
    {"action": "duplicate", "duplicate_of", "note"}, or
    {"action": "not_content"} -- the capture was an instruction directed at
    the tool itself (create/edit/delete/organize its own files), not
    something to file. Caught live: "make me a file that..." and "edit that
    file and add..." were both filed as nonsense notes about themselves
    before this check existed.
    Never raises -- falls back to a single safe "create" item in the "concept"
    type on any failure (bad JSON, connection drop, malformed response)."""
    if _is_meta_command(capture_text):
        return [{"action": "not_content"}]
    duplicate_of = _check_duplicate(capture_text, candidates)
    if duplicate_of:
        return [{"action": "duplicate", "duplicate_of": duplicate_of, "note": ""}]
    return _atomize(capture_text, candidates)
