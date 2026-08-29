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
partially match a title. If the instruction names something not in the list, or could plausibly
mean more than one note without covering ALL of them, respond "unclear" -- do not guess which
note(s) it means.

Produce "delete_all" when the instruction clearly means EVERY note, or every note of one type,
rather than any specific one -- "delete all notes", "delete everything", "delete all that is
here", "wipe the vault", "clear it all out", "delete all my logs". This is a real, answerable
instruction, NOT an unclear one: "all" is precise about scope even though it names no title.
Set "note_type" to "concept", "project", "entity", or "log" when the instruction limits itself
to one of those kinds ("delete all my logs" -> "log"); set it to null when it means everything.

CRITICAL: "delete_all" means literally EVERY note (or every note of one named type) and nothing
less. If the instruction QUALIFIES which notes -- "the old ones", "the useless ones", "the
duplicates", "the ones about X", "the ones I don't need" -- that is a SUBSET you have no way to
identify, and it is NOT delete_all. Deleting everything when the user meant a handful is the
worst mistake you can make here. Those go to "ask".

Produce "add_to" when the instruction asks to PUT SOME CONTENT INTO an existing note -- a URL,
a link, a fact, a detail, a reminder: "add this link to my X project", "put this under X", "save
this reference in X", "add that to X", "keep this in X". This is the action for content the user
wants stored inside a specific note, and it is very common -- do NOT answer "unclear" just
because the instruction also contains the content itself.
- "target" is the exact existing note title to add to, copied verbatim from the list.
- "text" is the actual CONTENT to store -- the URL, fact, or detail itself, with the
  instruction wording stripped out. For "https://example.com/x Add this link to my Y project",
  text is "https://example.com/x", NOT the whole sentence. Never put the instruction phrasing
  ("add this to", "keep this in") into "text".
- If the instruction refers to content from earlier in the conversation rather than restating it
  ("add the link there", "put that in X"), take the actual content from the recent conversation
  you are given and use it as "text". If you cannot recover the actual content, answer "unclear".

Do NOT use "link" for this. "link" is only for making one existing note reference ANOTHER
existing note; it cannot store a URL or any other text. If the instruction gives you content to
save rather than naming two existing notes, it is "add_to".

When the instruction is a real vault instruction but you cannot safely pin down exactly what it
means, produce "ask" with a short, specific question that would resolve it -- do NOT guess, and
do NOT give up. You are an assistant having a conversation, not a parser: asking is always
better than refusing. Use it for any instruction that names a SUBSET you cannot identify --
"delete the old notes", "delete the useless ones", "delete the duplicates", "clean up",
"organize my vault" -- for a named note that doesn't match anything in the list, and for
anything that could plausibly mean two or more different notes. A qualified subset is an "ask",
never a "delete_all": ask which ones they mean.
- Make the question answerable in one short reply. Name the real candidates when there are only
  a few ("Do you mean X or Y?"), and say what you'd be acting on.
- Never ask about something you already know; only ask for what's actually missing.

Reserve "unclear" for input that is not an actionable vault instruction at all and no answer
would fix.

You may be given recent conversation from this session. If the instruction uses "it"/"that"/
"the one I just mentioned"/"the same project we talked about" and the recent conversation
clearly identifies exactly one specific existing note being discussed, that counts as
unambiguously naming it -- resolve the reference to that note's exact title. If the recent
conversation doesn't clearly resolve it to one specific note, treat it as unclear rather than
guessing.

Respond with ONLY a JSON object, one of:
{"action": "delete", "target": str}
{"action": "delete_all", "note_type": str or null}
{"action": "link", "source": str, "target": str}
{"action": "add_to", "target": str, "text": str}
{"action": "ask", "question": str}
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

APPEND_TEXT_SYSTEM_PROMPT = """You are given something the user typed that is about to be saved
into one of their existing notes. Return the CONTENT worth storing, with the wording that was
addressed to the note-filing tool stripped out.

Strip only the request wrapper -- phrases like "add this to my X project", "keep this as a
reference in X", "put that in X", "save this", "note that", "remember". Those are how the user
told the tool what to do; they are not part of what they wanted written down.

Keep everything that is substance: URLs, facts, names, numbers, dates, technical details, and
any wording that carries meaning about the thing itself. Copy the substance VERBATIM -- never
summarize it, shorten it, rephrase it, correct its spelling, or add to it.

If the whole thing is substance, return it unchanged. When unsure whether a phrase is substance
or request wrapper, KEEP it: a few extra words in a note are harmless, losing information is
not.

Respond with ONLY a JSON object:
{"text": str}
"""

PLACEMENT_SYSTEM_PROMPT = """You decide WHERE in an existing note some new content belongs.

"body" -- it belongs to what the note IS. It defines, describes, or corrects the subject
itself: a reference URL for the project, a clarification of what the thing does, a detail that
a reader needs in order to understand the note at all. Someone reading the note top to bottom
should meet this immediately, not in a dated log at the bottom.

"updates" -- it belongs to what HAPPENED to the subject since. A change, a milestone, a new
capability, a status, a decision made on a date. It reads naturally as "on this date, this
became true" and would clutter the description if it sat in the body.

Prefer "body" for reference material and defining detail; prefer "updates" for events and
progress. When genuinely torn, choose "updates" -- it is the reversible, non-intrusive spot.

Respond with ONLY a JSON object:
{"placement": "body" or "updates"}
"""

RETITLE_SYSTEM_PROMPT = """You judge whether an existing note's title should be improved, given
what the note now contains.

Default to null. Renaming a note rewrites its filename and every link pointing at it, so it has
a real cost -- only worth paying when the current title is genuinely bad.

Genuinely bad means: cryptic or meaningless, a mangled fragment of a sentence, containing a
typo, padded with filler, or no longer describing what the note is actually about.

A good title is MINIMAL and UNDERSTANDABLE: the shortest phrase a person could read a year from
now and know what's inside. Usually 2-5 plain words, and always SHORTER than what it replaces --
if your proposal is longer than the current title, it is not an improvement, so return null
instead. Drop filler words ("new", "a", "the", "my", "for", "of"), dates, and padding like
"Notes on" / "Overview of". No trailing punctuation.

A title that is merely lowercase, hyphenated, or slug-like is NOT bad -- that's just how it was
written down, it still reads fine, and renaming it churns links for nothing. Leave it alone.
Never change what the note is ABOUT.

Respond with ONLY a JSON object:
{"title": str or null}
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
- title: MINIMAL and UNDERSTANDABLE -- the shortest phrase someone could read a year from now
  and know what's inside. Usually 2-5 plain words. Specific, not a broad category name, but
  never a mangled fragment of the user's sentence either. No dates, no filler, no "Notes on" /
  "Overview of" padding, no trailing punctuation.
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


def _format_clarification(clarification: str) -> str:
    """The user's answer to a question this tool just asked them. Folded into
    the command-parse prompt so the follow-up parse sees both the original
    instruction and the answer that disambiguated it, rather than trying to
    interpret a bare "the second one" on its own."""
    if not clarification:
        return ""
    return (
        f'\nYou asked the user to clarify, and they answered: "{clarification.strip()}"\n'
        f'Treat that answer as authoritative for resolving the instruction below. If it now\n'
        f'identifies what to act on, act -- do not ask again.\n'
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


def _call_command_parse(capture_text: str, known_titles: list[str], session_history: list[dict] = None,
                         clarification: str = None) -> dict:
    prompt = (
        f"Existing note titles:\n{_titles_block(known_titles)}\n"
        f"{_format_session_history(session_history)}"
        f"{_format_clarification(clarification)}\nInstruction:\n{capture_text}"
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


def _call_append_text(capture_text: str, target: dict) -> dict:
    prompt = (
        f'This is being saved into the note "{target["title"]}".\n\nThe user typed:\n{capture_text}'
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": APPEND_TEXT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _extract_append_text(capture_text: str, target: dict) -> str:
    """The substance of a capture, minus the instruction wrapper aimed at the
    tool. Reported live: "<url> THIS TO MY metadata extraction project" was
    filed with the instruction wording included, when only the URL was
    wanted.

    Verified rather than trusted: the result is rejected unless every word in
    it also appears in the original, which catches the real failure mode
    here (the model paraphrasing, summarizing, or "fixing" a typo instead of
    extracting) and falls back to the raw capture. Extraction can only ever
    lose information, so every failure path keeps the full text."""
    original = capture_text.strip()
    try:
        result = _call_append_text(original, target)
        if not isinstance(result, dict):
            return original
        extracted = _as_str(result.get("text"), "")
        if not extracted:
            return original
        original_words = set(_WORD_RE.findall(original.lower()))
        if not set(_WORD_RE.findall(extracted.lower())) <= original_words:
            return original  # rephrased/invented rather than extracted
        return extracted
    except Exception:
        return original


def _call_placement(text: str, target: dict) -> dict:
    prompt = (
        f'Note: "{target["title"]}" (type: {target.get("type", "concept")})\n'
        f'What it currently says: {target.get("excerpt", "")}\n\nNew content to place:\n{text}'
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": PLACEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


def _choose_placement(text: str, target: dict) -> str:
    """"body" or "updates". Defaults to "updates" on any failure -- a dated
    bullet at the bottom is the unintrusive choice, and the one this tool did
    unconditionally before it could decide at all."""
    try:
        result = _call_placement(text, target)
        if isinstance(result, dict) and _as_str(result.get("placement"), "") == "body":
            return "body"
    except Exception:
        pass
    return "updates"


def _call_retitle(target: dict, added_text: str) -> dict:
    prompt = (
        f'Current title: "{target["title"]}"\n'
        f'What the note says: {target.get("excerpt", "")}\n'
        f'Just added to it: {added_text}'
    )
    response = ollama.chat(
        model=config.OLLAMA_MODEL, format="json",
        messages=[
            {"role": "system", "content": RETITLE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response["message"]["content"])


MAX_TITLE_WORDS = 8
MIN_EXPANDABLE_WORDS = 4  # a 1-2 word cryptic title may grow this far, nothing else may grow


def _better_title(target: dict, added_text: str) -> str | None:
    """A genuinely better title for a note just written to, or None.

    Guarded rather than trusted: a retitle rewrites the filename and every
    inbound link, so a bad one is disruptive in a way an extra Updates line
    is not. Rejects anything empty, unchanged, or over-long -- and, since
    "minimal" is the stated bar, anything WORDIER than the title it would
    replace. Caught live: "github-repo-data-collection" (already fine) was
    "improved" to the longer "New GitHub Repo for Data Collection", which is
    churn, not a gain. A very short cryptic title is the one case where
    growing is legitimate, so titles of 1-2 words may expand up to
    MIN_EXPANDABLE_WORDS."""
    try:
        result = _call_retitle(target, added_text)
        if not isinstance(result, dict):
            return None
        title = _as_str(result.get("title"), "").strip().rstrip(".")
        if not title or _normalize_title(title) == _normalize_title(target["title"]):
            return None
        if len(title.split()) > MAX_TITLE_WORDS or len(title) > 80:
            return None
        current_words = len(_normalize_title(target["title"]).split())
        budget = max(current_words, MIN_EXPANDABLE_WORDS) if current_words <= 2 else current_words
        if len(title.split()) > budget:
            return None
        return title
    except Exception:
        return None


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
            # notes are all missing a title. Two notes sharing a fallback share
            # a filename, and filenames are now the identity of a note: the
            # second would be merged into the first rather than written as its
            # own file, silently collapsing two distinct concepts into one.
            # Index-suffix each fallback when needed.
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
                    session_history: list[dict] = None,
                    clarification: str = None) -> dict | None:
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
        result = _call_command_parse(capture_text, known_titles, session_history, clarification)
        if not isinstance(result, dict):
            return None
        action = str(result.get("action", "")).lower()

        if action == "delete":
            target = _find_by_title(_as_str(result.get("target"), ""), known_notes)
            if not target or not _confirm_delete(capture_text, target, session_history):
                return None
            return {"action": "delete", "target_path": target["path"], "target_title": target["title"]}

        if action == "delete_all":
            # Deliberately NOT gated behind _confirm_delete's LLM vote: an
            # unverifiable model opinion is the wrong guard for an operation
            # whose scope is already unambiguous. The caller prompts the
            # human with the exact file list instead -- a real confirmation
            # from the person whose vault it is beats three model votes.
            note_type = _as_str(result.get("note_type"), "").lower()
            note_type = note_type if note_type in VALID_TYPES else None
            targets = [
                n for n in known_notes
                if note_type is None or n.get("type", "concept") == note_type
            ]
            if not targets:
                return None
            return {"action": "delete_all", "note_type": note_type, "targets": targets}

        if action == "ask":
            question = _as_str(result.get("question"), "")
            return {"action": "ask", "question": question} if question else None

        if action == "add_to":
            target = _find_by_title(_as_str(result.get("target"), ""), known_notes)
            text = _as_str(result.get("text"), "")
            if not target or not text:
                return None
            # Purely additive (same "## Updates" insertion as an automatic
            # append), so no confirmation vote: the worst case is one extra
            # line in a note, which the user can see and undo. Confirmation
            # votes are reserved for operations that destroy content.
            return {
                "action": "append", "target_path": target["path"],
                "target_title": target["title"], "text": text,
            }

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


def _append_item(capture_text: str, target: dict) -> dict:
    """Build the append action: what to store, where in the note to put it,
    and whether writing it makes the note's title worth improving."""
    text = _extract_append_text(capture_text, target)
    return {
        "action": "append",
        "target_path": target["path"],
        "target_title": target["title"],
        "text": text,
        "placement": _choose_placement(text, target),
        "new_title": _better_title(target, text),
    }


def process_capture(capture_text: str, candidates: list[dict], known_notes: list[dict] = None,
                     session_history: list[dict] = None, clarification: str = None) -> list[dict]:
    """Returns a list of items, each one of:
    {"action": "create", "title", "type", "tags", "body", "links"},
    {"action": "append", "target_path", "target_title", "text"} -- a small
    update to an existing note's subject, merged into it rather than
    spawning a new file. Reported directly: three near-duplicate files
    ("Idea Agent Project", "...- Smart Feature", "...(2)") accumulated from
    captures that were really just new facts about the same project.
    {"action": "duplicate", "duplicate_of", "note"},
    {"action": "delete", "target_path", "target_title"},
    {"action": "delete_all", "note_type", "targets"} -- every note, or every
    note of one type. Unlike single-note delete this is NOT gated on an LLM
    confirmation vote: its scope is already unambiguous, so there's nothing
    for a model to verify; the caller prompts the human with the real file
    list instead. Or
    {"action": "link", "source_path", "source_title", "target_path", "target_title"}
    -- an unambiguous, confirmed vault-management instruction, or
    {"action": "ask", "question"} -- a real vault instruction that couldn't
    be pinned down safely, so it asks instead of guessing OR refusing. The
    caller is expected to put the question to the user and call again with
    their reply as `clarification`. Requested directly: "make sure that the
    ai knows he is an ai not just a note taker so let him ask me questions
    if not sure". Only one round -- a second unresolved pass refuses, or
    {"action": "not_content"} -- not an actionable vault instruction at all,
    and no answer would fix it. Caught live: "make me a file that..." and
    "edit that file and add..." were both filed as nonsense notes about
    themselves before instruction-detection existed.

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
        command = _parse_command(capture_text, known_notes, session_history, clarification)
        # One question per capture. If the answer still didn't resolve it,
        # refuse rather than starting an interrogation -- the user came here
        # to save a thought, not to play twenty questions.
        if command and command["action"] == "ask" and clarification:
            return [{"action": "not_content"}]
        return [command] if command else [{"action": "not_content"}]
    duplicate_of = _check_duplicate(capture_text, candidates, session_history)
    if duplicate_of:
        # Duplicate-detection is an OR-ensemble (any one "yes" wins), which
        # is right for catching restatements but over-fires on a capture that
        # merely SHARES A SUBJECT with an existing note. Reported live: a
        # reference URL the user explicitly wanted saved into his project
        # note ("<url> Keep this as reference in my ... project") was judged
        # a duplicate of that note and dropped -- it survived only as a
        # truncated line in the daily log. The README's claim that these
        # false positives are harmless ("just logs and links") held only
        # while nothing could be lost by one; it can, so verify before
        # dropping. Unanimous-redundant to discard, so a single vote saying
        # "this adds something new" is enough to keep it -- the whole point
        # of this tool is not losing ideas.
        dup_note = next((c for c in candidates if c["title"] == duplicate_of), None)
        if dup_note and not _is_redundant_update(capture_text, dup_note, session_history):
            return [_append_item(capture_text, dup_note)]
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
        return [_append_item(capture_text, append_target)]
    return _atomize(capture_text, candidates, session_history)
