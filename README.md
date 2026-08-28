# Idea Agent

A local terminal agent that files your raw ideas into an Obsidian vault for you.

Type an idea into the REPL. The agent embeds it, searches your existing vault notes for
anything related using RAG (retrieval-augmented generation), then asks a local LLM to turn
the raw idea into a structured note: a title, a folder, tags, an expanded body, and links to
the related notes it found. It writes that note straight into your vault, reusing an existing
folder when one fits and creating a new one only when nothing does.

Everything runs locally: the LLM is a local [Ollama](https://ollama.com) model, and the
embeddings are a local `sentence-transformers` model. No API keys, no data leaving your
machine.

## How it works

1. You type an idea at the `idea>` prompt.
2. The idea is embedded and compared against a local index of your vault's existing notes
   (cosine similarity over a plain JSON file -- no external vector database).
3. A local Ollama model (`qwen2.5:7b-instruct` by default) receives the idea, your vault's
   current top-level folders, and the related notes found in step 2, and returns a structured
   `{title, folder, tags, body, related_titles}`.
4. The agent writes `<vault>/<folder>/<title>.md` with YAML frontmatter and `[[wikilinks]]`
   to the related notes. Obsidian's own backlink pane picks up the reverse links automatically.
5. The new note gets added to the local index so future ideas can find it too.

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

- Type anything to file it as a new idea.
- `/list` -- show recently filed ideas.
- `/help` -- show the command list.
- `/quit` -- exit.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

149 tests, no live Ollama server or network access required -- the embedding model and
`ollama.chat` are mocked/faked for speed and determinism.

## Configuration

All settings live in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `VAULT_PATH` | *(required)* | Path to your Obsidian vault |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model used to classify and write notes |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model for embeddings |
| `TOP_K` | `5` | Max number of related notes to retrieve per idea |
