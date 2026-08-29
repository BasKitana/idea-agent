# Knowledge Agent

A local terminal agent that turns raw captures (ideas, thoughts, meeting notes) into a
structured Obsidian vault, automatically -- and that you can also directly command to manage
that vault (link, update, delete specific existing notes by name).

Type a capture into the REPL. The agent checks whether you've already written about it, checks
whether it's really just a small update to something you already have, splits it into distinct
atomic notes if it covers more than one concept, classifies each into one of four types, and
files it with proper frontmatter and links -- all locally, no API keys. Naming an exact
existing note in an instruction ("delete X", "link X to Y") gets executed if unambiguous;
vague vault instructions ("clean up my vault") are refused rather than guessed at.

Everything runs on-device: the LLM is a local [Ollama](https://ollama.com) model, embeddings
are a local `sentence-transformers` model, and the retrieval index is a plain JSON file. No
data leaves your machine.

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
  vote made the miss rate *worse*, not better -- a false positive there is harmless (just logs
  and links), so biasing toward catching it is free. The instruction-detection and delete
  checks use the opposite: unanimous vote, because a false positive there either silently
  discards real content or deletes a real note -- both worse than asking again. Delete
  additionally gets its own independent re-confirmation pass on top of the initial parse.
- **Folders are a closed 4-way classification, never free LLM text.** The vault mandates
  exactly `01_Concepts/`, `02_Projects/`, `03_Entities/`, `04_Logs/`. This also closes off an
  entire bug class from an earlier free-folder version of this tool (path traversal via a
  malicious/malformed folder name, and everything defaulting to "Inbox").
- **Command targets must match a real existing title exactly (punctuation/case aside), never
  fuzzy or semantic matching.** Delete and link only ever act on a title the model copied
  verbatim from the vault's real note list; case/hyphen/spacing differences are normalized
  away, but a name that doesn't resolve to a real note is refused, not guessed at.
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
- `/list` -- show recently filed notes.
- `/help` -- show the command list.
- `/quit` / `/exit` -- exit.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

230 tests, no live Ollama server or network access required -- the embedding model and
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
