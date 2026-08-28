import sys

import ollama
from rich.console import Console
from rich.panel import Panel

import agent
import config
import vault
from rag import RagIndex

console = Console()

HELP_TEXT = """[bold]Commands[/bold]
  /help          show this message
  /list          show recently filed ideas
  /quit, /exit   exit
Anything else you type is filed as a new idea."""


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


def process_idea(idea_text: str, index: RagIndex):
    related = index.query(idea_text)
    result = agent.classify_idea(idea_text, vault.list_folders(config.VAULT_PATH), related)
    path = vault.write_note(
        config.VAULT_PATH, result["folder"], result["title"],
        result["tags"], result["body"], result["related_titles"],
    )
    try:
        index.add(path, result["title"], f"{result['title']}\n{result['body']}")
    except Exception:
        console.print("[yellow]Filed, but the RAG index update failed -- it'll resync next launch.[/yellow]")

    body = f"[bold]{result['title']}[/bold]\nFolder: {result['folder']}\nFile: {path}"
    if result["related_titles"]:
        body += "\nLinked: " + ", ".join(result["related_titles"])
    console.print(Panel(body, title="Filed", border_style="green"))


def list_recent(index: RagIndex, n: int = 10):
    entries = sorted(index.data.items(), key=lambda kv: kv[1].get("mtime", 0), reverse=True)[:n]
    if not entries:
        console.print("[dim]No ideas filed yet.[/dim]")
        return
    for key, entry in entries:
        console.print(f"  {entry['title']}  [dim]({key})[/dim]")


def main():
    error = check_ollama()
    if error:
        console.print(Panel(error, title="Ollama not ready", border_style="red"))
        sys.exit(1)

    console.print(Panel(
        f"Vault: {config.VAULT_PATH}\nModel: {config.OLLAMA_MODEL}",
        title="Idea Agent", border_style="cyan",
    ))

    index = RagIndex()
    with console.status("Indexing existing notes..."):
        index.sync()

    console.print(HELP_TEXT)

    while True:
        try:
            text = console.input("\n[bold cyan]idea>[/bold cyan] ").strip()
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
                    process_idea(text, index)
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("[dim]Bye.[/dim]")


if __name__ == "__main__":
    main()
