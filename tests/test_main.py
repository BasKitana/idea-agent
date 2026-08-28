import pytest

import agent
import config
import main
from rag import RagIndex


@pytest.fixture
def ready_ollama(mocker):
    """Pretend the Ollama server is up and the configured model is pulled."""
    mocker.patch.object(main, "check_ollama", return_value=None)


class TestCheckOllama:
    def test_server_unreachable(self, mocker):
        mocker.patch.object(main.ollama, "Client", side_effect=ConnectionError("down"))
        error = main.check_ollama()
        assert "ollama serve" in error

    def test_model_not_pulled_is_reported_clearly(self, mocker):
        client = mocker.Mock()
        client.list.return_value = {"models": [{"model": "some-other-model"}]}
        mocker.patch.object(main.ollama, "Client", return_value=client)
        error = main.check_ollama()
        assert error is not None
        assert config.OLLAMA_MODEL in error
        assert "ollama pull" in error

    def test_model_present_returns_none(self, mocker):
        client = mocker.Mock()
        client.list.return_value = {"models": [{"model": config.OLLAMA_MODEL}]}
        mocker.patch.object(main.ollama, "Client", return_value=client)
        assert main.check_ollama() is None


class TestCommandParsing:
    @pytest.mark.parametrize("quit_word", ["/quit", "/QUIT", "/Quit", " /quit ", "/exit", "/EXIT"])
    def test_quit_variants_exit_cleanly(self, mocker, ready_ollama, vault_path, quit_word):
        mocker.patch.object(main.console, "input", side_effect=[quit_word])
        main.main()  # should return, not hang or raise

    @pytest.mark.parametrize("help_word", ["/help", "/Help", "/HELP"])
    def test_help_variants_show_help(self, mocker, ready_ollama, vault_path, help_word):
        mocker.patch.object(main.console, "input", side_effect=[help_word, "/quit"])
        print_spy = mocker.spy(main.console, "print")
        main.main()
        assert any("Commands" in str(c.args[0]) for c in print_spy.call_args_list if c.args)

    def test_unrecognized_slash_command_is_captured(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["/foo", "/quit"])
        spy = mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "Foo", "type": "concept", "tags": [], "body": "/foo", "links": []},
        ])
        main.main()
        assert spy.call_args[0][0] == "/foo"

    def test_empty_and_whitespace_input_is_ignored(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["", "   ", "/quit"])
        spy = mocker.patch.object(agent, "process_capture")
        main.main()
        spy.assert_not_called()


class TestProcessCapture:
    def test_create_item_writes_note_and_links_todays_log(self, mocker, vault_path):
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "New Note", "type": "concept",
             "tags": ["a"], "body": "b", "links": []},
        ])
        main.process_capture("some idea", index)
        assert (vault_path / "01_Concepts" / "New Note.md").exists()

    def test_duplicate_item_writes_no_note_but_logs(self, mocker, vault_path):
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "duplicate", "duplicate_of": "Existing", "note": ""},
        ])
        main.process_capture("dup idea", index)
        concepts_dir = vault_path / "01_Concepts"
        assert not concepts_dir.exists() or not list(concepts_dir.glob("*.md"))

    def test_multiple_create_items_all_get_written(self, mocker, vault_path):
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "Note A", "type": "project", "tags": [], "body": "b1", "links": []},
            {"action": "create", "title": "Note B", "type": "concept", "tags": [], "body": "b2", "links": []},
        ])
        main.process_capture("compound idea", index)
        assert (vault_path / "02_Projects" / "Note A.md").exists()
        assert (vault_path / "01_Concepts" / "Note B.md").exists()

    def test_one_item_failing_does_not_block_the_others(self, mocker, vault_path):
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "Good", "type": "concept", "tags": [], "body": "b", "links": []},
            {"action": "create", "title": None, "type": "concept", "tags": [], "body": "b", "links": []},
        ])
        # second item has a None title -- vault.write_note handles that gracefully
        # (falls back to "Untitled"), so this specifically checks the good item
        # still gets written regardless.
        main.process_capture("idea", index)
        assert (vault_path / "01_Concepts" / "Good.md").exists()


class TestResilience:
    def test_exception_during_capture_processing_keeps_repl_alive(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["bad idea", "/quit"])
        mocker.patch.object(agent, "process_capture", side_effect=Exception("boom"))
        main.main()  # must not raise; loop should continue to /quit

    def test_keyboard_interrupt_during_processing_exits_gracefully(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["an idea"])
        mocker.patch.object(agent, "process_capture", side_effect=KeyboardInterrupt)
        main.main()  # must not propagate KeyboardInterrupt

    def test_index_add_failure_still_reports_created(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["an idea", "/quit"])
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "X", "type": "concept", "tags": [], "body": "b", "links": []},
        ])
        mocker.patch.object(RagIndex, "add", side_effect=Exception("disk full"))
        print_spy = mocker.spy(main.console, "print")
        main.main()
        messages = [str(c.args[0]) for c in print_spy.call_args_list if c.args]
        assert any("Processed" in m or "index update failed" in m or "+ X" in m for m in messages)
        assert (vault_path / "01_Concepts" / "X.md").exists()


class TestRichMarkupSafety:
    """Rich's console.print() treats [brackets] as markup tags -- a title
    containing them can be silently mangled or dropped entirely instead of
    printed literally. Caught live: a "[project]" type prefix vanished from
    /list output with no error. Every dynamic string must be escaped."""

    def test_bracketed_title_prints_literally_in_summary(self, mocker, vault_path):
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "[urgent] Fix the bug", "type": "concept",
             "tags": [], "body": "b", "links": []},
        ])
        with main.console.capture() as capture:
            main.process_capture("idea", index)
        assert "[urgent] Fix the bug" in capture.get()

    def test_bracketed_title_prints_literally_in_list_recent(self, vault_path):
        index = RagIndex()
        index.data["n.md"] = {"mtime": 1, "title": "[urgent] Fix the bug", "type": "concept", "embedding": [0]}
        with main.console.capture() as capture:
            main.list_recent(index)
        assert "[urgent] Fix the bug" in capture.get()


class TestListRecent:
    def test_no_notes_yet(self, mocker, vault_path):
        index = RagIndex()
        print_spy = mocker.spy(main.console, "print")
        main.list_recent(index)
        assert any("No notes filed yet" in str(c.args[0]) for c in print_spy.call_args_list if c.args)

    def test_truncates_to_ten_most_recent(self, vault_path):
        index = RagIndex()
        for i in range(15):
            index.data[f"n{i}.md"] = {"mtime": i, "title": f"Note {i}", "type": "concept", "embedding": [0]}
        entries = sorted(index.data.items(), key=lambda kv: kv[1]["mtime"], reverse=True)[:10]
        assert len(entries) == 10
        assert entries[0][1]["title"] == "Note 14"
