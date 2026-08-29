# Knowledge Agent

**The goal: capture your ideas and keep them ordered, without losing them along the way.**

Two ways ideas get lost. You don't write them down, because filing them properly is friction
in the moment. Or you do write them down, and they dissolve into a pile of near-duplicate,
disconnected files you'll never find again. This tool exists to remove both -- you type the
idea and stop thinking about it; it decides where the idea belongs in your Obsidian vault and
puts it there, connected to what you already know.

It works like a clerk, not a note-taker: fewer files, better placed. A new idea about something
you've already written gets merged into that existing note instead of becoming file number
four on the same subject. A genuinely new idea gets its own note, classified, tagged, and
linked to the related things it came from. Nothing is silently dropped, and nothing is
overwritten -- updates are additive, deletes go to the Recycle Bin, and anything too ambiguous
to place safely is handed back to you rather than guessed at.

Type a capture into the REPL and it checks whether you've already written about it, whether
it's really an update to something you have, whether it covers more than one distinct subject,
classifies each note into one of four types, and files it with proper frontmatter and links.
You can also just tell it what to do -- "delete X", "link X to Y", "delete all my logs" --
and it executes when the target is unambiguous, asks first when the action is destructive, and
refuses when the instruction is too vague to act on safely.

Everything runs on-device: the LLM is a local [Ollama](https://ollama.com) model, embeddings
are a local `sentence-transformers` model, and the retrieval index is a plain JSON file. No
API keys, no accounts, no data leaving your machine.

## Design

Built after researching real Obsidian PKM methodology (community consensus on folders vs
tags vs links, atomicity, MOC/graveyard failure modes) and verifying exact Obsidian 1.13+
mechanics (frontmatter parsing, wikilink resolution, one-directional backlinks) against the
official docs, plus empirically measuring how this pipeline actually behaves against the
local model at every step rather than assuming it. Several design decisions came directly out
of that measurement, not out of guessing:

- **Every yes/no judgment is its own focused LLM call, never folded into a richer generation
  schema.** A single call asked to both judge duplication AND generate notes in one response
  missed an exact-match duplicate almost every time, confirmed on two different model sizes --
  a prompt-structure problem, not a capacity one. The same split now covers duplicate
  detection, "is this an update to something existing", "is this an instruction to the tool",
  and delete confirmation.
- **The voting bias direction depends on which mistake is worse, and that was measured per
  check, not assumed.** Duplicate-detection uses an OR-ensemble (any positive vote counts):
  measured directly, the model's per-call "yes" rate is well under 50%, and a 2-of-3 majority
  vote made the miss rate *worse*, not better, so biasing toward catching it is right. That
  bias was originally justified by calling its false positives harmless; they turned out not to
  be (see the duplicate-verdict bullet below), so the ensemble still fires freely but no longer
  gets the last word on discarding anything. The instruction-detection and delete
  checks use the opposite: unanimous vote, because a false positive there either silently
  discards real content or deletes a real note -- both worse than asking again. Delete
  additionally gets its own independent re-confirmation pass on top of the initial parse.
- **Folders are a closed 4-way classification, never free LLM text.** The vault mandates
  exactly `01_Concepts/`, `02_Projects/`, `03_Entities/`, `04_Logs/`. This also closes off an
  entire bug class from an earlier free-folder version of this tool (path traversal via a
  malicious/malformed folder name, and everything defaulting to "Inbox").
- **"Put this in that note" needed to be sayable, and a duplicate verdict must never silently
  drop content.** Reported live, one session hit both halves of this. A reference URL captured
  as "&lt;url&gt; Keep this as reference in my ... project" was judged a duplicate of that very
  project note and dropped -- it survived only as a truncated line in the daily log. Then every
  attempt to fix it by hand ("No I mean add the link there", "Add this link to the same project
  we talked about") was recognized as an instruction and refused, because the command vocabulary
  had no way to express it: `link` only makes one note reference another, it cannot store a URL
  or any other text. Two fixes. An `add_to` action now stores given content in a named note,
  recovering the content from session history when the instruction only refers back to it
  ("add the link there"); it takes no confirmation vote, since it is purely additive and those
  votes are reserved for operations that destroy something. And a duplicate verdict is now
  verified before anything is discarded -- the same unanimous redundancy check the append path
  uses, so a single vote saying "this adds new information" is enough to keep it. Duplicate
  detection's OR-ensemble was only ever safe on the assumption that a false positive costs
  nothing; it can cost real content, so it no longer gets the last word. Measured across the
  exact failing phrasings: 0 refusals in 9 runs, and "No I mean add the link there" now resolves
  correctly 3/3.
- **"All" is a precise scope, not a vague one -- and it gets a human confirmation, not an LLM
  vote.** Reported live: "delete all that is here" and "delete all existing notes" were both
  correctly recognized as instructions and then refused anyway, because the command layer could
  only express "delete &lt;one exact title&gt;". Bulk delete now parses into its own action. It's
  deliberately the one destructive path with no LLM confirmation vote: those votes exist to
  check "did I identify the right single note", and an all-scoped delete has nothing of the
  sort to verify -- so the guard is the person seeing the real file list and typing "yes" in
  full. Genuinely fuzzy scopes ("delete the old notes", "clean up") are still refused, since
  those describe a judgment call rather than a scope. Measured 21/21 across both phrasings,
  type-scoped deletes, the vague controls, and single-title delete.
- **Command targets must match a real existing title exactly (punctuation/case aside), never
  fuzzy or semantic matching.** Delete and link only ever act on a title the model copied
  verbatim from the vault's real note list; case/hyphen/spacing differences are normalized
  away, but a name that doesn't resolve to a real note is refused, not guessed at.
- **Fewer files by default: the tool behaves like a clerk consolidating related material, not
  a note-taker giving every fact its own page.** Two mechanisms drive this. Atomization now
  keeps every fact/detail/sub-point about ONE subject in that subject's single note, splitting
  into separate notes only when a capture genuinely covers multiple, actually-different
  subjects -- not just multiple parts of the same one. And the append check now defaults to
  merging same-subject content into the existing note regardless of size (previously it only
  merged "small" updates and pushed anything substantial into its own new, related-but-separate
  file) -- it only refuses to merge when the new content is clearly about a different,
  standalone subject. Measured live across 6 trials of a multi-fact same-subject capture: 5/6
  correctly consolidated into a single note (previously this would have split every time by
  design); the one miss still produced exactly one file, just via two atomized notes instead of
  one.
- **Broadening append to merge substantial content introduced a new failure mode, caught by
  testing the change rather than assuming it was safe.** Once append no longer required a "small"
  update, a reworded restatement of a fact the note already had (missed by duplicate-detection,
  a pre-existing, already-documented imperfection) started getting appended as if it were new --
  polluting the note with a repeated line instead of just spawning an extra file. Bolting an
  "is this actually new information" clause onto the append-decision prompt itself did not
  reliably catch it (measured: an obvious acronym-expansion restatement, "RAG" to "retrieval-
  augmented generation", got through 0 times fixed by that alone). The actual fix followed this
  codebase's own established rule -- split into its own focused, single-question call rather
  than combining judgments -- as a dedicated, UNANIMOUS-vote "is this redundant with the note
  it's about to join" check that runs right before an append executes; a redundant verdict
  downgrades the outcome to a duplicate (logged and linked, nothing written) instead of an
  append. Unanimous, not OR, because suppressing real new content is the costly mistake here
  (same direction as delete-confirm and instruction-detection), the opposite of duplicate-
  detection's OR-ensemble. This closed the common case (verified live) but not a specific
  acronym-expansion phrasing, which is a systematic model bias rather than per-call noise --
  measured 0/8 across repeated trials, so more voting rounds cannot fix it; documented here as a
  known residual limitation rather than claimed as solved.
- **The append decision needed the same architecture as duplicate-detection, not a score
  pre-filter.** The first version of append-broadening still required exactly one candidate to
  score >= AUTO_LINK_SCORE before it would even ask the LLM whether to append -- with 0 or 2+
  candidates crossing that bar, the judgment call never ran at all, and reported live, this was
  the common case, not the rare one: it looked like the tool "just kept adding files" regardless
  of what was typed. Rebuilt to mirror `_check_duplicate` exactly -- every retrieved candidate
  (no score gate) goes into one call that reads through all of them and picks which, if any, the
  capture belongs with. Verified live against the exact real-vault sequence that used to split
  ("the idea agent project now is smart" / "the idea agent now knows how to use Obsidian."): 3/3
  trials now merge into one file, while a genuinely unrelated second capture in the same session
  still correctly produces a separate note.
- **Session memory fixes retrieval, not just judgment -- carrying the prior turn's subject
  forward as a candidate, not just adding conversation text to the prompt.** Feeding recent
  turns into the LLM's prompts alone fixed pronoun resolution for commands ("delete it" against
  a note just discussed, confirmed live: 5/5 correct with history vs. 0/5 without), but a
  pronoun-heavy follow-up like "it also uses Ollama" still failed to append to the note it
  clearly referred to -- because candidate retrieval is pure embedding similarity on the raw
  text, and a sentence built almost entirely of pronouns doesn't embed close enough to score as
  a candidate at all, regardless of what the LLM is told. Fixed by carrying the previous turn's
  subject note forward as a guaranteed candidate every turn, independent of its embedding
  score.

Routine filing (duplicates, small updates) never edits or deletes a note -- duplicates get
logged to that day's log note with a link to what already covers it, updates get appended
under an "## Updates" heading without touching anything already there. Delete and link are the
only operations that act on existing notes at all, and only when explicitly, unambiguously
instructed to.

## How it works

0. On launch, the RAG index is synced against every note already in the vault -- so the agent
   starts each session already knowing everything you've filed, not just what happens in that
   session. On top of that, a short-term memory of the current session's own turns (capped at
   `SESSION_MEMORY_SIZE`, never persisted to disk) is fed into every check so later captures
   like "delete it" or "it also does X" can resolve what "it" refers to.
1. You type a capture at the `capture>` prompt.
2. If it's an instruction naming existing note(s) unambiguously (delete, link), it's executed
   directly -- delete requires an independent re-confirmation pass first, and always goes to
   the OS recycle bin, never a permanent delete. A vague instruction is refused with an
   explanation rather than guessed at.
3. Otherwise it's embedded and compared against the local index of existing notes.
4. A duplicate check runs against the closest matches. If it's genuinely the same idea as an
   existing note, nothing new is written -- it's logged to today's log note with a link to
   the existing note instead.
5. If there's exactly one strongly related existing note, a separate check asks whether this
   capture is just a small new fact/update about that same subject. If so, it's appended under
   an "## Updates" heading in that note instead of spawning a near-duplicate file.
6. Otherwise, the capture is atomized: one note per distinct concept (a vague one-liner stays
   exactly one note -- the model was observed inventing fictional sub-topics from short input
   until this was explicitly guarded against). A capture strongly tied to one existing note
   gets an explicit hint naming that note's type, since type classification was observed
   defaulting to the wrong type even with the related note already visible in context.
7. Each note is written with YAML frontmatter (`title`, `date`, `type`, `tags`) into the
   folder its type maps to, with `[[wikilinks]]` to genuinely related notes -- auto-linked to
   any strongly related note regardless of whether the model's own output mentioned it, since
   that was measured to be unreliable on its own. Obsidian computes backlinks itself, so links
   are only ever written in one direction.
8. Every new note also links to today's log note, so nothing is ever a true orphan with zero
   inbound links.
9. An `Index.base` file (Obsidian's core Bases feature, no plugin needed) gets refreshed at
   the vault root every run for browsing by type.

## Setup

Requirements: Python 3.10+, [Ollama](https://ollama.com) installed.

```
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
```

Copy `.env.example` to `.env` and set `VAULT_PATH` to your Obsidian vault's folder:

```
VAULT_PATH=C:\path\to\your\Obsidian\Vault
```

## Running it

Start the Ollama server if it isn't already running, then run the agent:

```
ollama serve
python main.py
```

On Windows, `run.bat` does both in one step -- double-click it (or make a desktop shortcut
to it) to launch the agent without touching a terminal.

## REPL commands

- Type an idea, thought, or fact to capture, atomize, and file it.
- Name an existing note exactly to delete or link it ("delete X", "link X to Y") -- executed
  if unambiguous, refused with an explanation otherwise.
- Store something in a specific note ("add this link to my X project", "save this reference in
  X", "put that in X") -- appended under that note's "## Updates" heading. If you refer back to
  something instead of restating it ("add the link there"), it recovers the content from earlier
  in the session.
- Delete in bulk ("delete all notes", "delete everything", "delete all my logs") -- prints the
  exact file list and waits for you to type "yes" before touching anything.
- `/list` -- show recently filed notes.
- `/help` -- show the command list.
- `/quit` / `/exit` -- exit.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

256 tests, no live Ollama server or network access required -- the embedding model and
`ollama.chat` are mocked/faked for speed and determinism.

## Configuration

All settings live in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `VAULT_PATH` | *(required)* | Path to your Obsidian vault |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model used for dedup, atomization, and commands |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model for embeddings |
| `DEDUP_RETRIEVE_SCORE` | `0.40` | Minimum cosine similarity to pull a note in as a dedup/relation candidate |
| `DEDUP_TOP_K` | `8` | Max candidates retrieved per capture |
| `ATOMIZE_MIN_WORDS` | `12` | Below this word count, atomization is forced to exactly one note |
| `SESSION_MEMORY_SIZE` | `10` | Recent session turns kept in short-term memory for resolving "it"/"that" |
| `AUTO_LINK_SCORE` | `0.50` | Minimum similarity for auto-linking/append/type-hint (a single dominant related note) |
