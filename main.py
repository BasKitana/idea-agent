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
Anything else you type is captured, atomized, and filed. Naming an exact existing note
("delete X", "link X to Y") is executed if unambiguous; vague vault instructions are
refused rather than guessed at."""


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
    known_notes = [
        {"title": e["title"], "path": k, "type": e.get("type", "concept"), "excerpt": e.get("excerpt", "")}
        for k, e in index.data.items()
    ]
    items = agent.process_capture(capture_text, candidates, known_notes)

    today = date.today().isoformat()
    created = []       # (title, path, links)
    duplicates = []    # (capture_text, existing_title)
    updated = []       # (capture_text, target_title)
    deleted = []       # (title,)
    linked = []        # (source_title, target_title)
    failed = []        # (title_or_desc, error)
    not_content = False

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
            elif item["action"] == "append":
                path = vault.append_update(config.VAULT_PATH, item["target_path"], item["text"])
                try:
                    index.add(path, item["target_title"], path.read_text(encoding="utf-8"))
                except Exception:
                    console.print(f"[yellow]Updated '{item['target_title']}', but the RAG "
                                  "index update failed -- it'll resync next launch.[/yellow]")
                updated.append((capture_text, item["target_title"]))
            elif item["action"] == "delete":
                vault.delete_note(config.VAULT_PATH, item["target_path"])
                index.data.pop(item["target_path"], None)
                index.save()
                deleted.append(item["target_title"])
            elif item["action"] == "link":
                path = vault.add_link(config.VAULT_PATH, item["source_path"], item["target_title"])
                try:
                    index.add(path, item["source_title"], path.read_text(encoding="utf-8"))
                except Exception:
                    console.print(f"[yellow]Linked, but the RAG index update failed -- it'll "
                                  "resync next launch.[/yellow]")
                linked.append((item["source_title"], item["target_title"]))
            elif item["action"] == "duplicate":
                duplicates.append((capture_text, item["duplicate_of"]))
            elif item["action"] == "not_content":
                not_content = True
        except Exception as e:
            failed.append((item.get("title") or item.get("target_title") or item.get("duplicate_of") or "?", e))

    if created or duplicates or updated:
        vault.append_daily_log_links(
            config.VAULT_PATH, [t for t, _, _ in created], duplicates, updated,
        )

    _print_summary(created, duplicates, failed, not_content, updated, deleted, linked)


def _print_summary(created, duplicates, failed, not_content=False, updated=None, deleted=None, linked=None):
    # Titles/paths/errors are LLM-generated or user-supplied text and may
    # contain literal [brackets] -- console.print() treats those as markup
    # tags, so unescaped dynamic text can be silently mangled or dropped
    # (caught live: a bracketed type prefix vanished entirely). Escape every
    # dynamic value before interpolating it into a Rich-printed string.
    lines = []
    for title, path, note_type in created:
        lines.append(f"[bold green]+ {escape(title)}[/bold green] "
                      f"[dim]({escape(note_type)} -> {escape(str(path))})[/dim]")
    for raw, target in (updated or []):
        lines.append(f"[bold cyan]~ updated[/bold cyan] -> [[{escape(target)}]]")
    for source, target in (linked or []):
        lines.append(f"[bold cyan]~ linked[/bold cyan] [[{escape(source)}]] -> [[{escape(target)}]]")
    for title in (deleted or []):
        lines.append(f"[bold magenta]- deleted[/bold magenta] [[{escape(title)}]] "
                      f"[dim](sent to recycle bin)[/dim]")
    for raw, existing in duplicates:
        lines.append(f"[bold yellow]= already covered[/bold yellow] -> "
                      f"[[{escape(existing)}]]")
    for desc, err in failed:
        lines.append(f"[bold red]x {escape(str(desc))}: {escape(str(err))}[/bold red]")
    if not_content:
        lines.append("[dim]That reads like an instruction I can't safely act on -- it either "
                      "wasn't specific about which existing note(s) it means, or it named "
                      "something that doesn't exist. Name a note exactly, or type an idea, "
                      "thought, or fact instead.[/dim]")

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
