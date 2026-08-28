# Knowledge Agent

A local terminal agent that turns raw captures (ideas, thoughts, meeting notes) into a
structured Obsidian vault, automatically.

Type a capture into the REPL. The agent checks whether you've already written about it,
splits it into distinct atomic notes if it covers more than one concept, classifies each into
one of four types, and files it with proper frontmatter and links -- all locally, no API keys.

Everything runs on-device: the LLM is a local [Ollama](https://ollama.com) model, embeddings
are a local `sentence-transformers` model, and the retrieval index is a plain JSON file. No
data leaves your machine.

## Design

Built after researching real Obsidian PKM methodology (community consensus on folders vs
tags vs links, atomicity, MOC/graveyard failure modes) and verifying exact Obsidian 1.13+
mechanics (frontmatter parsing, wikilink resolution, one-directional backlinks) against the
official docs, plus empirically measuring how this pipeline actually behaves against the
local model rather than assuming it -- three design decisions came directly out of that:

- **Dedup is two separate LLM calls, not one.** A single call asked to both judge duplication
  and generate notes in one schema missed an exact-match duplicate almost every time --
  confirmed on two different model sizes, so it's a prompt-structure problem, not a capacity
  one. Splitting "is this a duplicate?" into its own focused yes/no call fixed it.
- **The duplicate check is an OR-ensemble of 3 calls, not a majority vote.** Measured
  directly: the model's per-call "yes" rate is well under 50% (a systematic bias toward "no"),
  so requiring 2-of-3 agreement made the miss rate *worse*. Accepting any single positive vote
  raised the catch rate substantially with zero measured false positives on a
  related-but-distinct control case.
- **Folders are a closed 4-way classification, never free LLM text.** The vault mandates
  exactly `01_Concepts/`, `02_Projects/`, `03_Entities/`, `04_Logs/`. This also closes off an
  entire bug class from an earlier free-folder version of this tool (path traversal via a
  malicious/malformed folder name, and everything defaulting to "Inbox").

Duplicates never edit or delete an existing note -- they're logged to that day's log note
with a link to what already covers it, so nothing is ever silently lost either direction.

## How it works

1. You type a capture at the `capture>` prompt.
2. It's embedded and compared against the local index of existing notes.
3. A duplicate check runs against the closest matches. If it's genuinely the same idea as an
   existing note, nothing new is written -- it's logged to today's log note with a link to
   the existing note instead.
4. Otherwise, the capture is atomized: one note per distinct concept (a vague one-liner stays
   exactly one note -- the model was observed inventing fictional sub-topics from short input
   until this was explicitly guarded against).
5. Each note is written with YAML frontmatter (`title`, `date`, `type`, `tags`) into the
   folder its type maps to, with `[[wikilinks]]` to genuinely related notes. Obsidian computes
   backlinks itself, so links are only ever written in one direction.
6. Every new note also links to today's log note, so nothing is ever a true orphan with zero
   inbound links.
7. An `Index.base` file (Obsidian's core Bases feature, no plugin needed) gets refreshed at
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

- Type anything to capture, atomize, and file it.
- `/list` -- show recently filed notes.
- `/help` -- show the command list.
- `/quit` / `/exit` -- exit.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

157 tests, no live Ollama server or network access required -- the embedding model and
`ollama.chat` are mocked/faked for speed and determinism.

## Configuration

All settings live in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `VAULT_PATH` | *(required)* | Path to your Obsidian vault |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model used for dedup and atomization |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model for embeddings |
| `DEDUP_RETRIEVE_SCORE` | `0.40` | Minimum cosine similarity to pull a note in as a dedup/relation candidate |
| `DEDUP_TOP_K` | `8` | Max candidates retrieved per capture |
| `ATOMIZE_MIN_WORDS` | `12` | Below this word count, atomization is forced to exactly one note |
