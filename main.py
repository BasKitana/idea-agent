import sys
from datetime import date

import ollama
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

import agent
import bases
import config
import vault
from rag import RagIndex

console = Console()

HELP_TEXT = """[bold]Commands[/bold]
  /help          show this message
  /list          show recently filed notes
  /quit, /exit   exit
Anything else you type is captured, atomized, and filed."""


def check_ollama() -> str | None:
    """Returns None when ready, or an error message to show the user."""
    try:
        models = ollama.Client(host=config.OLLAMA_HOST).list()
    except Exception:
        return f"Can't reach Ollama at {config.OLLAMA_HOST}.\nStart it with: ollama serve"

    available = {m["model"] for m in models["models"]}
    if config.OLLAMA_MODEL not in available:
        return f"Model '{config.OLLAMA_MODEL}' isn't pulled yet.\nRun: ollama pull {config.OLLAMA_MODEL}"
    return None


def process_capture(capture_text: str, index: RagIndex):
    candidates = index.query(
        capture_text, top_k=config.DEDUP_TOP_K, min_score=config.DEDUP_RETRIEVE_SCORE,
    )
    items = agent.process_capture(capture_text, candidates)

    today = date.today().isoformat()
    created = []       # (title, path, links)
    duplicates = []    # (capture_text, existing_title)
    failed = []        # (title_or_desc, error)

    for item in items:
        try:
            if item["action"] == "create":
                links = list(dict.fromkeys(item["links"] + [today]))  # anti-orphan: always link today's log
                path = vault.write_note(
                    config.VAULT_PATH, item["type"], item["title"],
                    item["tags"], item["body"], links,
                )
                try:
                    index.add(path, item["title"], f"{item['title']}\n{item['body']}", note_type=item["type"])
                except Exception:
                    console.print(f"[yellow]Filed '{item['title']}', but the RAG index update "
                                  "failed -- it'll resync next launch.[/yellow]")
                created.append((item["title"], path, item["type"]))
            elif item["action"] == "duplicate":
                duplicates.append((capture_text, item["duplicate_of"]))
        except Exception as e:
            failed.append((item.get("title") or item.get("duplicate_of") or "?", e))

    if created or duplicates:
        vault.append_daily_log_links(
            config.VAULT_PATH, [t for t, _, _ in created], duplicates,
        )

    _print_summary(created, duplicates, failed)


def _print_summary(created, duplicates, failed):
    # Titles/paths/errors are LLM-generated or user-supplied text and may
    # contain literal [brackets] -- console.print() treats those as markup
    # tags, so unescaped dynamic text can be silently mangled or dropped
    # (caught live: a bracketed type prefix vanished entirely). Escape every
    # dynamic value before interpolating it into a Rich-printed string.
    lines = []
    for title, path, note_type in created:
        lines.append(f"[bold green]+ {escape(title)}[/bold green] "
                      f"[dim]({escape(note_type)} -> {escape(str(path))})[/dim]")
    for raw, existing in duplicates:
        lines.append(f"[bold yellow]= already covered[/bold yellow] -> "
                      f"[[{escape(existing)}]]")
    for desc, err in failed:
        lines.append(f"[bold red]x {escape(str(desc))}: {escape(str(err))}[/bold red]")

    if not lines:
        console.print("[dim]Nothing came of that.[/dim]")
        return
    console.print(Panel("\n".join(lines), title="Processed", border_style="cyan"))


def list_recent(index: RagIndex, n: int = 10):
    entries = sorted(index.data.items(), key=lambda kv: kv[1].get("mtime", 0), reverse=True)[:n]
    if not entries:
        console.print("[dim]No notes filed yet.[/dim]")
        return
    for key, entry in entries:
        note_type = escape(str(entry.get("type", "?")))
        title = escape(str(entry.get("title", key)))
        console.print(f"  {note_type}: {title}  [dim]({escape(key)})[/dim]")


def main():
    error = check_ollama()
    if error:
        console.print(Panel(error, title="Ollama not ready", border_style="red"))
        sys.exit(1)

    console.print(Panel(
        f"Vault: {config.VAULT_PATH}\nModel: {config.OLLAMA_MODEL}",
        title="Knowledge Agent", border_style="cyan",
    ))

    index = RagIndex()
    with console.status("Indexing existing notes..."):
        index.sync()
    try:
        bases.write_index(config.VAULT_PATH)
    except Exception:
        pass  # the Bases index is a convenience, never worth blocking startup over

    console.print(HELP_TEXT)

    while True:
        try:
            text = console.input("\n[bold cyan]capture>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue
        command = text.lower()
        if command in ("/quit", "/exit"):
            break

        try:
            if command == "/help":
                console.print(HELP_TEXT)
            elif command == "/list":
                list_recent(index)
            else:
                with console.status("Thinking..."):
                    process_capture(text, index)
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("[dim]Bye.[/dim]")


if __name__ == "__main__":
    main()
