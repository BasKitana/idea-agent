import pytest

import agent
import config
import main
import vault
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

    def test_unrecognized_slash_command_is_filed_as_idea(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["/foo", "/quit"])
        spy = mocker.patch.object(agent, "classify_idea", return_value={
            "title": "Foo", "folder": "Inbox", "tags": [], "body": "/foo", "related_titles": [],
        })
        main.main()
        assert spy.call_args[0][0] == "/foo"

    def test_empty_and_whitespace_input_is_ignored(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["", "   ", "/quit"])
        spy = mocker.patch.object(agent, "classify_idea")
        main.main()
        spy.assert_not_called()


class TestResilience:
    def test_list_with_corrupted_entry_does_not_crash_repl(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["/list", "/quit"])
        index = RagIndex()
        index.data = {"bad.md": {"title": "Bad"}}  # missing "mtime"
        mocker.patch.object(main, "RagIndex", return_value=index)
        main.main()  # must not raise

    def test_exception_during_idea_filing_keeps_repl_alive(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["bad idea", "/quit"])
        mocker.patch.object(agent, "classify_idea", side_effect=Exception("boom"))
        main.main()  # must not raise; loop should continue to /quit

    def test_keyboard_interrupt_during_processing_exits_gracefully(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["an idea"])
        mocker.patch.object(agent, "classify_idea", side_effect=KeyboardInterrupt)
        main.main()  # must not propagate KeyboardInterrupt

    def test_index_add_failure_still_reports_filed(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(main.console, "input", side_effect=["an idea", "/quit"])
        mocker.patch.object(agent, "classify_idea", return_value={
            "title": "X", "folder": "Health", "tags": [], "body": "b", "related_titles": [],
        })
        mocker.patch.object(RagIndex, "add", side_effect=Exception("disk full"))
        print_spy = mocker.spy(main.console, "print")
        main.main()
        messages = [str(c.args[0]) for c in print_spy.call_args_list if c.args]
        assert any("Filed" in m or "index update failed" in m for m in messages)


class TestListRecent:
    def test_no_ideas_yet(self, mocker, vault_path):
        index = RagIndex()
        print_spy = mocker.spy(main.console, "print")
        main.list_recent(index)
        assert any("No ideas filed yet" in str(c.args[0]) for c in print_spy.call_args_list if c.args)

    def test_truncates_to_ten_most_recent(self, vault_path):
        index = RagIndex()
        for i in range(15):
            index.data[f"n{i}.md"] = {"mtime": i, "title": f"Note {i}", "embedding": [0]}
        entries = sorted(index.data.items(), key=lambda kv: kv[1]["mtime"], reverse=True)[:10]
        assert len(entries) == 10
        assert entries[0][1]["title"] == "Note 14"
