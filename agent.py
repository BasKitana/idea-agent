import json
import re

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
ALWAYS content, even when phrased with "build"/"make"/"create"/"should support"/"should add"/
"should have" -- those verbs and phrasings alone do NOT mean it's an instruction, including
when they sound imperative. A feature idea or roadmap item for the user's OWN project ("X
should support Y", "we should add Z") is content describing that project, not a command to
this tool. Only classify as an instruction when the ACTION is explicitly aimed at the tool's
own notes/files/folders/vault, not at some other project or system.

You are given the vault's EXISTING note titles below. If the input names one or more of those
EXACT existing titles and describes doing something to them -- deleting, linking, connecting,
merging -- that is always an instruction, even if the phrasing itself sounds like a neutral
statement (e.g. "link A to B" naming two real note titles is an instruction, not content).

You may also be given recent conversation from this session. A short input like "delete it" or
"add more to that" is an instruction if the recent conversation makes clear what "it"/"that"
refers to -- treat resolved pronouns the same as an explicit existing title.

Examples of instructions (is_instruction: true) -- all explicitly about the tool's own notes/files:
"Make me a file that links to another file." "Edit that note and add X." "Delete the old
notes." "Create an empty folder [in the vault]." "Organize my vault." "Make a project folder
with no notes inside [the vault]." "Link the Idea Agent note to the Claude Code note." "Delete
the Old Test Note." "Connect X and Y." -- an instruction to link, connect, or delete two or
more notes that already exist in the vault is ALWAYS an instruction, even with no other
imperative-sounding language, since it's an action on the tool's own files by definition.

Examples of genuine content (is_instruction: false) -- these describe projects/ideas to build,
plan, or remember, NOT the tool's own files, even though they sound imperative:
"I want to build a RAG pipeline using ChromaDB." "Build an app that tracks water intake."
"A research project where I collect data from a GitHub repository." "Create a marketing plan
for the launch." "Vercel's edge functions look good for the auth layer." "Remember to follow
up with Sarah about Clerk." "Idea Agent should support voice input as a new feature." "The API
should support pagination for large result sets." "We should add rate limiting to the auth
endpoint."

Respond with ONLY a JSON object:
{"is_instruction": true or false}
"""

COMMAND_SYSTEM_PROMPT = """You parse an instruction directed at a note-filing tool's vault into
one specific, safe action. You are given the instruction and a list of the vault's EXISTING
note titles -- this is the complete list; no other notes exist.

Only produce "delete" or "link" when the instruction unambiguously names note(s) from that
EXACT list. Copy the title(s) verbatim from the list -- never invent, guess, abbreviate, or
partially match a title. If the instruction is vague ("delete the old notes", "clean up",
"organize my vault"), names something not in the list, or could plausibly mean more than one
note, respond "unclear" -- do not guess which note(s) it means.

You may be given recent conversation from this session. If the instruction uses "it"/"that"/
"the one I just mentioned" and the recent conversation clearly identifies exactly one specific
existing note being discussed, that counts as unambiguously naming it -- resolve the pronoun to
that note's exact title. If the recent conversation doesn't clearly resolve it to one specific
note, treat it as unclear rather than guessing.

Respond with ONLY a JSON object, one of:
{"action": "delete", "target": str}
{"action": "link", "source": str, "target": str}
{"action": "unclear"}
"""

DELETE_CONFIRM_SYSTEM_PROMPT = """You are a final safety check before permanently removing a
note from someone's personal knowledge base. You are given the user's instruction and the
specific note that a separate matching step has ALREADY identified as the target -- that
matching is done and correct; small wording differences between the instruction and the
note's exact title (spacing, hyphens, capitalization -- e.g. "HNSW indexing" vs
"HNSW-indexing") are normal and do NOT indicate a wrong match. Do not re-judge whether the
title matches; only judge intent.

If the instruction uses a pronoun ("delete it", "remove that"), you may be given recent
conversation from this session explaining what it refers to -- that resolution is ALSO
already done and correct; do not doubt it just because the instruction itself doesn't name
the note explicitly. Judge only whether deleting is really what's meant, given that context.

Answer yes ONLY if the instruction's INTENT clearly and specifically means to delete this
note's subject. Answer no if there is genuine doubt about the intent itself -- e.g. the
instruction could mean something other than deletion, or could plausibly refer to a
different note's subject entirely. When genuinely uncertain about intent, answer no -- a
missed deletion just means the user asks again; a wrong deletion loses their content.

Respond with ONLY a JSON object:
{"confirmed": true or false}
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

APPEND_SYSTEM_PROMPT = """You judge whether a new capture should be APPENDED as an update to ONE
of a list of existing notes, rather than becoming its own separate new note. This tool acts as a
clerk that consolidates related material into as few files as possible -- not a note-taker that
gives every new fact its own page. Read through ALL the existing notes given below before
deciding; the right match is not always the first or most obviously worded one.

Answer with a note's EXACT title, copied verbatim, whenever the capture is genuinely ABOUT that
note's SAME subject AND adds information the note doesn't already have -- a new fact, detail,
capability, decision, or status update. Default to appending same-subject content even when it's
substantial; do not withhold it just because there's a lot to say, and do not skip it just
because more than one note is loosely related -- pick whichever ONE note the capture is most
specifically about.

Answer null when any of these is true:
- No existing note shares the capture's subject -- it's a different, standalone topic that
  deserves its own new note.
- The capture doesn't add anything new to the best-matching note -- it's just restating or
  rewording a fact that note already covers (a duplicate, not an update).
- More than one note is a plausible, comparably strong match and it's genuinely unclear which
  one this belongs to (uncommon -- most captures have one clearly best-fitting note even among
  several related ones).

Respond with ONLY a JSON object:
{"append_to": str or null}
"""

REDUNDANT_UPDATE_SYSTEM_PROMPT = """You judge whether a new capture is REDUNDANT with an
existing note it's about to be appended to -- i.e. it restates a fact the note already states,
in different words, without adding anything new.

Answer redundant=true ONLY when the capture and the note's existing content are clearly saying
the exact same fact, just reworded (e.g. an acronym spelled out, a synonym swapped in) -- no new
detail, number, decision, or capability beyond what the note already says.

Answer redundant=false whenever the capture adds ANY new detail, fact, decision, capability, or
status the note doesn't already have -- even if it also restates something already there
alongside the new part. When genuinely uncertain whether it's truly redundant or actually adds
something new, answer redundant=false.

Respond with ONLY a JSON object:
{"redundant": true or false}
"""

ATOMIZE_SYSTEM_PROMPT = """You decompose a raw capture into atomic notes for a technical
Obsidian vault. The capture has already been confirmed to NOT duplicate any existing note.

NOTE TYPES (choose exactly one per note):
- concept: general, reusable technical/academic knowledge that is NOT tied to one specific
  project of the user's -- e.g. "how HNSW indexing works", "the difference between REST and
  GraphQL". If it only makes sense in the context of one of the user's own named
  projects/systems, it is NOT a concept, even if it sounds technical.
- project: an active development task, roadmap, system specification, or ANY update, detail,
  feature, or capability added to one of the user's own specific projects. If the capture is
  clearly about a project the user already has -- it names that project, or you were given it
  as a strongly related existing note -- default to "project", not "concept".
- entity: a specific API, organization, person, or distinct external component/tool.
- log: a chronological record of something that HAPPENED -- an event, a meeting, a specific
  moment in time. Not every mention of a project is a log entry; only use log when the capture
  describes an event occurring, not a fact/feature/detail about the project itself.

ATOMICITY (bias toward FEWER notes -- this tool acts as a clerk consolidating related material,
not a note-taker giving every fact its own page):
- If the capture describes ONE distinct subject, produce exactly ONE note, even if that subject
  has several facts, details, or sub-points -- put them all in that one note's body rather than
  splitting each point into its own note. A project's tech choices, features, and status are
  all part of that ONE project's note, not separate notes.
- Only split into multiple notes when the capture genuinely covers two or more SEPARATE
  subjects that don't share one overarching topic (e.g. a project update AND an unrelated
  reminder about a person, or two different projects mentioned in passing) -- one note per
  actual subject, linked to each other. Do not split just because a subject has multiple
  parts; split only when there are multiple, actually-different subjects.
- If the capture is vague, short, or a placeholder with no real distinguishable topic, produce
  exactly ONE note that captures it as-is. NEVER fabricate sub-topics, structure, or
  elaboration not actually present in the input. Inventing content the user didn't say is a
  serious failure, worse than under-splitting.

You may also be given RELATED notes (not duplicates, just related) -- link to them by exact
title when a note you create is genuinely, specifically connected to one.

You may also be given recent conversation from this session. If the raw capture uses a pronoun
or reference ("it", "that", "the one I just mentioned") use the recent conversation to resolve
what it refers to and write the note about that actual subject -- do not leave a vague pronoun
in the title or body.

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


def _relation_hint(candidates: list[dict]) -> str:
    """When there's one dominant strongly-related candidate, name its type
    explicitly rather than leaving the model to infer relevance from a
    generic candidate list. Measured directly: type classification for a
    capture strongly tied to an existing project scattered across
    concept/project/log (5/8, 1/8, 2/8) even though the related project note
    was already right there in the candidate list -- the same "secondary
    signal gets dropped" pattern behind the auto-link fix, just for type
    instead of links. Reuses AUTO_LINK_SCORE since it's the same "is this
    genuinely the same context" question."""
    strong = [c for c in candidates if c.get("score", 0) >= config.AUTO_LINK_SCORE]
    if len(strong) != 1:
        return ""
    c = strong[0]
    return (
        f"\nThis capture is strongly related to an existing note: \"{c['title']}\" "
        f"(type: {c.get('type', 'concept')}). If this capture describes an update, detail, "
        f"feature, or capability of that SAME project/entity, use its same type -- do not "
        f"default to \"concept\" just because it's easier. Only use a different type if this "
        f"capture clearly describes a distinct event (log) or a genuinely separate, general "
        f"idea unrelated to that specific note.\n"
    )


def _titles_block(known_titles: list[str]) -> str:
    return "\n".join(f"- {t}" for t in known_titles) or "(vault is empty, no notes exist)"


def _format_session_history(session_history: list[dict]) -> str:
    """Short-term memory: the last few turns of THIS REPL session, so a
    capture like "delete it" or "add more to that" can resolve what "it"/
    "that" refers to. Pronouns carry almost no embedding signal, so RAG
    retrieval alone can't do this -- it needs the actual recent conversation.
    Empty when there's no history yet (start of session, or nothing relevant
    happened), which is the common case and adds nothing to the prompt."""
    if not session_history:
        return ""
    lines = "\n".join(f'- You said: "{t["capture"]}" -> {t["summary"]}' for t in session_history)
    return (
        f'\nRecent conversation in this session (most recent last) -- use this to resolve '
        f'references like "it"/"that"/"the one I just mentioned" in the current input:\n{lines}\n'
    )


def _call_meta_check(capture_text: str, known_titles: list[str], session_history: list[dict] = None) -> dict:
    prompt = (
        f"Existing note titles:\n{_titles_block(known_titles)}\n"
        f"{_format_session_history(session_history)}\nInput:\n{capture_text}"
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": META_COMMAND_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _call_command_parse(capture_text: str, known_titles: list[str], session_history: list[dict] = None) -> dict:
    prompt = (
        f"Existing note titles:\n{_titles_block(known_titles)}\n"
        f"{_format_session_history(session_history)}\nInstruction:\n{capture_text}"
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": COMMAND_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _call_delete_confirm(capture_text: str, target: dict, session_history: list[dict] = None) -> dict:
    prompt = (
        f'Instruction:\n{capture_text}\n\nIdentified target note: "{target["title"]}" '
        f'(type: {target.get("type", "concept")}): {target.get("excerpt", "")}\n'
        f'{_format_session_history(session_history)}'
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": DELETE_CONFIRM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _call_append_check(capture_text: str, candidates: list[dict], session_history: list[dict] = None) -> dict:
    prompt = (
        f"Existing notes:\n{_format_candidates(candidates)}\n"
        f"{_format_session_history(session_history)}\nNew capture:\n{capture_text}"
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": APPEND_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _call_redundant_check(capture_text: str, target: dict, session_history: list[dict] = None) -> dict:
    prompt = (
        f'Existing note: "{target["title"]}": {target.get("excerpt", "")}\n'
        f'{_format_session_history(session_history)}\nNew capture:\n{capture_text}'
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": REDUNDANT_UPDATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _call_duplicate_check(capture_text: str, candidates: list[dict], session_history: list[dict] = None) -> dict:
    prompt = (
        f"Existing notes:\n{_format_candidates(candidates)}\n"
        f"{_format_session_history(session_history)}\nNew capture:\n{capture_text}"
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": DUPLICATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _call_atomize(capture_text: str, candidates: list[dict], session_history: list[dict] = None) -> dict:
    prompt = (
        f"Related notes (not duplicates):\n{_format_candidates(candidates)}\n"
        f"{_relation_hint(candidates)}"
        f"{_format_session_history(session_history)}"
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


def _one_meta_vote(capture_text: str, known_titles: list[str], session_history: list[dict]) -> bool:
    try:
        result = _call_meta_check(capture_text, known_titles, session_history)
        return isinstance(result, dict) and bool(result.get("is_instruction", False))
    except Exception:
        return False


def _is_meta_command(capture_text: str, known_titles: list[str] = None,
                      session_history: list[dict] = None, votes: int = 3) -> bool:
    """True if the capture is an instruction directed at the tool (create,
    edit, delete, organize) rather than genuine content to record.

    Needs known_titles: without it, "link A to B" naming two real existing
    notes has no way to be told apart from a neutral technical statement --
    measured directly, it was missed 5/5 with the capture text alone,
    fixed once the check could see A and B are actual vault note titles.

    Unanimous vote, not majority or OR -- the opposite bias from duplicate
    detection. There, a false positive is harmless (just logs+links), so
    OR-ensemble (favor catching it) was right. Here, a false positive
    silently DISCARDS real content the user typed -- worse than a false
    negative, which just creates a slightly-odd note (recoverable). Measured
    directly: genuine feature-idea phrasing ("X should support Y") sits at a
    true ~50% per-call rate for the model -- not a bias, a real coin flip --
    while actual instructions ("make me a file...", "delete the old notes")
    are near-100% per-call. Unanimous-3 exploits exactly that gap: costs
    nothing on the clear cases, and it's very hard for a genuinely ambiguous
    case to hit 3-for-3 by chance. Verified: 0/6 false positives across 4
    ambiguous feature-idea phrasings, 6/6 correct on 5 clear instructions.
    """
    known_titles = known_titles or []
    return all(_one_meta_vote(capture_text, known_titles, session_history) for _ in range(votes))


def _one_duplicate_vote(capture_text: str, candidates: list[dict], candidate_titles: set,
                         session_history: list[dict]) -> str | None:
    try:
        result = _call_duplicate_check(capture_text, candidates, session_history)
        if not isinstance(result, dict):
            return None
        target = _as_str(result.get("duplicate_of"), "")
        return target if target in candidate_titles else None
    except Exception:
        return None


def _check_duplicate(capture_text: str, candidates: list[dict],
                      session_history: list[dict] = None, votes: int = 3) -> str | None:
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
        vote = _one_duplicate_vote(capture_text, candidates, candidate_titles, session_history)
        if vote:
            return vote
    return None


def _one_append_vote(capture_text: str, candidates: list[dict], candidate_titles: set,
                      session_history: list[dict]) -> str | None:
    try:
        result = _call_append_check(capture_text, candidates, session_history)
        if not isinstance(result, dict):
            return None
        target = _as_str(result.get("append_to"), "")
        return target if target in candidate_titles else None
    except Exception:
        return None


def _one_redundant_vote(capture_text: str, target: dict, session_history: list[dict]) -> bool:
    try:
        result = _call_redundant_check(capture_text, target, session_history)
        return isinstance(result, dict) and bool(result.get("redundant", False))
    except Exception:
        return False


def _is_redundant_update(capture_text: str, target: dict, session_history: list[dict],
                          votes: int = 3) -> bool:
    """True only if EVERY vote agrees the capture adds nothing beyond what
    the target note already says. Unanimous, not OR -- the opposite bias
    from duplicate-detection, because this decides whether to SUPPRESS
    content rather than whether to flag it. A missed redundant line just
    means one repeated sentence gets appended (minor, easily fixed by hand);
    wrongly calling real new information "redundant" silently drops it
    forever, which is worse. Needs its own focused call rather than folding
    "is this new information" into the append-decision prompt -- measured
    directly: bolting that question onto APPEND_SYSTEM_PROMPT still let an
    obvious reworded restatement ("RAG note filing tool" -> "retrieval-
    augmented generation tool that files notes") through as a fresh append,
    the same "combined judgment degrades reliability" failure this codebase
    already hit once with duplicate-detection-plus-generation."""
    return all(_one_redundant_vote(capture_text, target, session_history) for _ in range(votes))


def _check_append(capture_text: str, candidates: list[dict],
                   session_history: list[dict] = None, votes: int = 3) -> dict | None:
    """Returns the candidate to append to, or None. Mirrors
    _check_duplicate's architecture exactly: every retrieved candidate is
    offered to ONE judgment call that picks which one (if any) fits, rather
    than pre-filtering to a single "dominant" candidate by score before the
    LLM ever gets a say.

    Reported live against the real vault: the old version required exactly
    one candidate scoring >= AUTO_LINK_SCORE before it would even ask the
    append question -- with 0 or 2+ candidates clearing that bar, which is
    common given how much embedding scores for related-but-differently-
    worded captures overlap (see DEDUP_RETRIEVE_SCORE's own calibration
    note), the append judgment never ran at all. The tool looked like it
    "just kept adding files" no matter what, because most captures never
    reached a check that could have said otherwise.

    OR-ensemble, same bias as duplicate-detection: a false-positive append
    still writes the capture's full text verbatim into a real, existing note
    -- just possibly not the ideal one -- which is recoverable and not data
    loss, unlike a missed append, which is what silently keeps producing
    extra files."""
    if not candidates:
        return None
    candidate_titles = {c["title"] for c in candidates}
    for _ in range(votes):
        target_title = _one_append_vote(capture_text, candidates, candidate_titles, session_history)
        if target_title:
            return next(c for c in candidates if c["title"] == target_title)
    return None


def _atomize(capture_text: str, candidates: list[dict], session_history: list[dict] = None) -> list[dict]:
    candidate_titles = {c["title"] for c in candidates}
    fallback_title = capture_text.strip()[:60] or "Untitled"

    for _ in range(2):
        try:
            result = _call_atomize(capture_text, candidates, session_history)
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
            # Don't depend on the LLM to remember a strongly-related candidate:
            # measured directly, given one right there in the prompt, it only
            # populated "links" in 4/6 runs. The retrieval score is already
            # known before this call even runs, so auto-link anything strong
            # regardless of what the LLM did -- a floor under its judgment,
            # not a replacement. Scoped to the single-note case: a multi-note
            # split doesn't cleanly tell us which resulting note a
            # whole-capture-level score actually belongs to.
            if len(notes) == 1:
                strong = [c["title"] for c in candidates if c.get("score", 0) >= config.AUTO_LINK_SCORE]
                notes[0]["links"] = list(dict.fromkeys(notes[0]["links"] + strong))
            if notes:
                return notes
        except Exception:
            continue

    return [{
        "action": "create", "title": fallback_title, "type": "concept",
        "tags": ["unsorted"], "body": capture_text.strip(), "links": [],
    }]


def _normalize_title(title: str) -> str:
    """Case/punctuation-insensitive form for title matching, NOT fuzzy/partial
    matching -- the actual words still have to match exactly. Reported live:
    a note auto-titled "HNSW-indexing" (hyphenated) by atomize legitimately
    couldn't be found by a delete instruction naming it "HNSW indexing"
    (space) under exact match, even though no reasonable person would type
    the hyphen back. Punctuation is incidental formatting the atomize step
    chose, not part of the note's actual identity."""
    return re.sub(r"[\s_-]+", " ", title).strip().lower()


def _find_by_title(title: str, known_notes: list[dict]) -> dict | None:
    normalized = _normalize_title(title)
    for n in known_notes:
        if _normalize_title(n["title"]) == normalized:
            return n
    return None


def _confirm_delete(capture_text: str, target: dict, session_history: list[dict] = None,
                     votes: int = 3) -> bool:
    """Unanimous re-confirmation, independent of and in addition to the
    initial command parse -- delete is the one operation in this codebase
    with no recovery path inside the tool itself (only the OS recycle bin).
    Same bias as the meta-command check: a missed delete costs nothing (the
    user just asks again), a wrong delete costs real content, so this
    requires every vote to agree and any doubt anywhere defaults to no.

    Needs session_history for the same reason command-parsing does: reported
    live, "delete it" (pronoun) resolved correctly to the right note at parse
    time, but confirmation without the session context that justified that
    resolution had no way to verify it and refused every time -- it couldn't
    tell "it" meant anything at all, let alone this specific note."""
    for _ in range(votes):
        try:
            result = _call_delete_confirm(capture_text, target, session_history)
            if not (isinstance(result, dict) and bool(result.get("confirmed", False))):
                return False
        except Exception:
            return False
    return True


def _parse_command(capture_text: str, known_notes: list[dict],
                    session_history: list[dict] = None) -> dict | None:
    """Returns a ready-to-execute command dict, or None if the instruction
    was too vague/ambiguous to safely act on (caller should refuse it, same
    as before this existed). known_notes is the FULL list of {"title","path"}
    for every note in the vault -- targets must match one of these exactly;
    a name the model invents or only partially matches is never trusted.
    session_history lets "delete it"/"link that to X" resolve a pronoun to
    one specific recently-discussed title -- still exact-match after that,
    never fuzzy."""
    known_titles = [n["title"] for n in known_notes]
    try:
        result = _call_command_parse(capture_text, known_titles, session_history)
        if not isinstance(result, dict):
            return None
        action = str(result.get("action", "")).lower()

        if action == "delete":
            target = _find_by_title(_as_str(result.get("target"), ""), known_notes)
            if not target or not _confirm_delete(capture_text, target, session_history):
                return None
            return {"action": "delete", "target_path": target["path"], "target_title": target["title"]}

        if action == "link":
            source = _find_by_title(_as_str(result.get("source"), ""), known_notes)
            target = _find_by_title(_as_str(result.get("target"), ""), known_notes)
            if not source or not target or source["title"] == target["title"]:
                return None
            return {
                "action": "link", "source_path": source["path"], "source_title": source["title"],
                "target_path": target["path"], "target_title": target["title"],
            }
    except Exception:
        pass
    return None


def process_capture(capture_text: str, candidates: list[dict], known_notes: list[dict] = None,
                     session_history: list[dict] = None) -> list[dict]:
    """Returns a list of items, each one of:
    {"action": "create", "title", "type", "tags", "body", "links"},
    {"action": "append", "target_path", "target_title", "text"} -- a small
    update to an existing note's subject, merged into it rather than
    spawning a new file. Reported directly: three near-duplicate files
    ("Idea Agent Project", "...- Smart Feature", "...(2)") accumulated from
    captures that were really just new facts about the same project.
    {"action": "duplicate", "duplicate_of", "note"},
    {"action": "delete", "target_path", "target_title"} or
    {"action": "link", "source_path", "source_title", "target_path", "target_title"}
    -- an unambiguous, confirmed vault-management instruction, or
    {"action": "not_content"} -- the capture was an instruction, but too
    vague/ambiguous to safely act on (named no real note, could mean more
    than one, etc.) -- refused rather than guessed at. Caught live: "make
    me a file that..." and "edit that file and add..." were both filed as
    nonsense notes about themselves before instruction-detection existed.

    session_history is this REPL session's short-term memory (recent
    {"capture", "summary"} turns, oldest first) -- lets a pronoun-only
    capture like "delete it" or "add more to that" resolve against what was
    just discussed, which RAG retrieval alone can't do since pronouns carry
    almost no embedding signal. Not persisted across restarts; the vault
    itself is the long-term memory, this is only for the current session.

    Never raises -- falls back to a single safe "create" item in the "concept"
    type on any failure (bad JSON, connection drop, malformed response)."""
    known_notes = known_notes or []
    known_titles = [n["title"] for n in known_notes]
    if _is_meta_command(capture_text, known_titles, session_history):
        command = _parse_command(capture_text, known_notes, session_history)
        return [command] if command else [{"action": "not_content"}]
    duplicate_of = _check_duplicate(capture_text, candidates, session_history)
    if duplicate_of:
        return [{"action": "duplicate", "duplicate_of": duplicate_of, "note": ""}]
    append_target = _check_append(capture_text, candidates, session_history)
    if append_target:
        # A restated near-duplicate of the target note's own content can
        # slip past _check_duplicate (measured live: reworded technical
        # phrasing missed 3/3 duplicate votes) and, once append is scoped to
        # "same subject" rather than "small update", would otherwise get
        # written into the note as if it were new -- silently polluting it
        # with a repeated fact. Catch that here as a duplicate instead of an
        # append: no new file (same outcome as before), and no redundant
        # line added to the existing note either.
        if _is_redundant_update(capture_text, append_target, session_history):
            return [{"action": "duplicate", "duplicate_of": append_target["title"], "note": ""}]
        return [{
            "action": "append", "target_path": append_target["path"],
            "target_title": append_target["title"], "text": capture_text.strip(),
        }]
    return _atomize(capture_text, candidates, session_history)
