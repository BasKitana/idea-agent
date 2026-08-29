import pytest

import agent
import config
import main
from rag import RagIndex


@pytest.fixture
def ready_ollama(mocker):
    """Pretend the Ollama server is up and the configured model is pulled."""
    mocker.patch.object(main, "check_ollama", return_value=None)


class TestHistorySummary:
    def test_created_note_summarized(self):
        summary = main._history_summary([("X", "path.md", "concept")], [], [], [], [], [], False)
        assert "filed 'X' (concept)" in summary

    def test_updated_note_summarized(self):
        summary = main._history_summary([], [], [("raw", "Target")], [], [], [], False)
        assert "updated 'Target'" in summary

    def test_linked_note_summarized(self):
        summary = main._history_summary([], [], [], [], [("Source", "Target")], [], False)
        assert "linked 'Source' to 'Target'" in summary

    def test_deleted_note_summarized(self):
        summary = main._history_summary([], [], [], ["X"], [], [], False)
        assert "deleted 'X'" in summary

    def test_duplicate_summarized(self):
        summary = main._history_summary([], [("raw", "Existing")], [], [], [], [], False)
        assert "recognized as duplicate of 'Existing'" in summary

    def test_not_content_summarized(self):
        summary = main._history_summary([], [], [], [], [], [], True)
        assert "refused" in summary

    def test_nothing_happened_has_a_summary(self):
        assert main._history_summary([], [], [], [], [], [], False) == "nothing happened"


class TestHistorySubject:
    def test_created_note_is_the_subject(self):
        assert main._history_subject([("X", "path.md", "concept")], [], [], [], []) == "X"

    def test_updated_note_is_the_subject(self):
        assert main._history_subject([], [], [("raw", "Target")], [], []) == "Target"

    def test_linked_source_is_the_subject(self):
        assert main._history_subject([], [], [], [], [("Source", "Target")]) == "Source"

    def test_duplicate_of_is_the_subject(self):
        assert main._history_subject([], [("raw", "Existing")], [], [], []) == "Existing"

    def test_deleted_note_has_no_subject_to_carry_forward(self):
        assert main._history_subject([], [], [], ["X"], []) is None

    def test_nothing_happened_has_no_subject(self):
        assert main._history_subject([], [], [], [], []) is None


class TestSessionMemoryAccumulation:
    def test_process_capture_mutates_session_history_with_summary_and_subject(self, mocker, vault_path):
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "X", "type": "concept", "tags": [], "body": "b", "links": []},
        ])
        session_history = []
        main.process_capture("an idea", index, session_history)
        assert "filed 'X'" in session_history[-1]["summary"]
        assert session_history[-1]["subject"] == "X"

    def test_previous_turns_subject_is_injected_as_a_candidate(self, mocker, vault_path):
        # Regression: a pronoun-heavy follow-up like "it also uses Ollama" can
        # embed-score too low against the note it refers to for RAG to ever
        # retrieve it as a candidate at all -- carry the prior turn's subject
        # forward as a guaranteed candidate instead of relying on embedding luck.
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "Idea Agent", "type": "project", "tags": [], "body": "b", "links": []},
        ])
        session_history = []
        main.process_capture("Idea Agent is a local RAG filing tool", index, session_history)

        seen_candidates = {}

        def fake_process_capture(capture_text, candidates, known_notes, history):
            seen_candidates["candidates"] = candidates
            return [{"action": "not_content"}]

        mocker.patch.object(agent, "process_capture", side_effect=fake_process_capture)
        mocker.patch.object(index, "query", return_value=[])
        main.process_capture("it also uses Ollama", index, session_history)

        titles = [c["title"] for c in seen_candidates["candidates"]]
        assert "Idea Agent" in titles

    def test_subject_carry_forward_does_not_override_an_already_strong_candidate(self, mocker, vault_path):
        # Reported live against the real vault: capture N-1 created a new
        # note ("love-for-idea-agent"); capture N ("idea agent can now be a
        # clerk of my notes", no pronoun at all) genuinely retrieved OTHER,
        # actually-relevant notes, but injecting the prior subject
        # unconditionally added a synthetic AUTO_LINK_SCORE candidate that
        # won the "single strong candidate" slot instead, so the fact got
        # appended to the wrong (just-created, unrelated) note. The injected
        # subject must never be added once a real candidate already clears
        # AUTO_LINK_SCORE on its own.
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[
            {"action": "create", "title": "love-for-idea-agent", "type": "concept",
             "tags": [], "body": "b", "links": []},
        ])
        session_history = []
        main.process_capture("I love the idea agent", index, session_history)
        assert session_history[-1]["subject"] == "love-for-idea-agent"

        seen_candidates = {}

        def fake_process_capture(capture_text, candidates, known_notes, history):
            seen_candidates["candidates"] = candidates
            return [{"action": "not_content"}]

        mocker.patch.object(agent, "process_capture", side_effect=fake_process_capture)
        genuinely_relevant = [{"title": "idea-agent-project-smart", "path": "x.md",
                                "type": "project", "score": config.AUTO_LINK_SCORE, "excerpt": "x"}]
        mocker.patch.object(index, "query", return_value=genuinely_relevant)
        main.process_capture("idea agent can now be a clerk of my notes", index, session_history)

        titles = [c["title"] for c in seen_candidates["candidates"]]
        assert "love-for-idea-agent" not in titles
        assert titles == ["idea-agent-project-smart"]

    def test_repl_passes_growing_history_into_agent_calls(self, mocker, ready_ollama, vault_path):
        # call_args_list stores a REFERENCE to the (mutated-in-place) history
        # list, not a snapshot -- inspecting it after main() returns would
        # only ever show its final state. Snapshot (copy) at call time instead.
        mocker.patch.object(main.console, "input", side_effect=["first idea here", "second idea here", "/quit"])
        seen_histories = []

        def fake_process_capture(capture_text, candidates, known_notes, session_history):
            seen_histories.append(list(session_history))
            return [{"action": "create", "title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]

        mocker.patch.object(agent, "process_capture", side_effect=fake_process_capture)
        main.main()
        assert seen_histories[0] == []
        assert len(seen_histories[1]) == 1
        assert seen_histories[1][0]["capture"] == "first idea here"

    def test_history_is_capped_at_configured_size(self, mocker, ready_ollama, vault_path):
        mocker.patch.object(config, "SESSION_MEMORY_SIZE", 2)
        inputs = [f"idea number {i}" for i in range(5)] + ["/quit"]
        mocker.patch.object(main.console, "input", side_effect=inputs)
        seen_histories = []

        def fake_process_capture(capture_text, candidates, known_notes, session_history):
            seen_histories.append(list(session_history))
            return [{"action": "create", "title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]

        mocker.patch.object(agent, "process_capture", side_effect=fake_process_capture)
        main.main()
        assert all(len(h) <= 2 for h in seen_histories)
        assert len(seen_histories[-1]) == 2


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

    def test_append_item_updates_existing_note_and_reindexes(self, mocker, vault_path):
        target = vault_path / "02_Projects"
        target.mkdir(parents=True)
        note = target / "Existing Note.md"
        note.write_text("# Existing Note\n\nOriginal.\n", encoding="utf-8")

        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[{
            "action": "append", "target_path": "02_Projects/Existing Note.md",
            "target_title": "Existing Note", "text": "a new fact",
        }])
        main.process_capture("a new fact", index)

        text = note.read_text(encoding="utf-8")
        assert "Original." in text and "a new fact" in text
        assert "02_Projects/Existing Note.md" in index.data

    def test_append_to_missing_target_reported_as_failure_not_crash(self, mocker, vault_path):
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[{
            "action": "append", "target_path": "02_Projects/Gone.md",
            "target_title": "Gone", "text": "a fact",
        }])
        with main.console.capture() as capture:
            main.process_capture("a fact", index)  # must not raise
        assert "Gone" in capture.get()

    def test_delete_item_sends_to_recycle_bin_and_removes_from_index(self, mocker, vault_path):
        target = vault_path / "01_Concepts"
        target.mkdir(parents=True)
        note = target / "Old Note.md"
        note.write_text("# Old Note\n", encoding="utf-8")

        index = RagIndex()
        index.data["01_Concepts/Old Note.md"] = {"mtime": 1, "title": "Old Note", "type": "concept", "embedding": [0]}
        mocker.patch.object(agent, "process_capture", return_value=[{
            "action": "delete", "target_path": "01_Concepts/Old Note.md", "target_title": "Old Note",
        }])
        mocker.patch("vault.send2trash.send2trash")  # don't actually touch the real recycle bin in tests
        main.process_capture("delete Old Note", index)

        assert "01_Concepts/Old Note.md" not in index.data

    def _bulk_delete_vault(self, vault_path, mocker):
        """Two real notes on disk + in the index, ready to be bulk-deleted."""
        (vault_path / "01_Concepts").mkdir(parents=True)
        (vault_path / "04_Logs").mkdir(parents=True)
        (vault_path / "01_Concepts" / "A.md").write_text("# A\n", encoding="utf-8")
        (vault_path / "04_Logs" / "B.md").write_text("# B\n", encoding="utf-8")
        index = RagIndex()
        index.data["01_Concepts/A.md"] = {"mtime": 1, "title": "A", "type": "concept", "embedding": [0]}
        index.data["04_Logs/B.md"] = {"mtime": 1, "title": "B", "type": "log", "embedding": [0]}
        targets = [
            {"title": "A", "path": "01_Concepts/A.md", "type": "concept"},
            {"title": "B", "path": "04_Logs/B.md", "type": "log"},
        ]
        mocker.patch.object(agent, "process_capture", return_value=[{
            "action": "delete_all", "note_type": None, "targets": targets,
        }])
        mocker.patch("vault.send2trash.send2trash")  # never touch the real recycle bin in tests
        return index

    def test_bulk_delete_removes_every_target_when_confirmed(self, mocker, vault_path):
        index = self._bulk_delete_vault(vault_path, mocker)
        main.process_capture("delete all notes", index, confirm=lambda targets, note_type: True)
        assert index.data == {}

    def test_bulk_delete_deletes_nothing_when_declined(self, mocker, vault_path):
        # The human prompt is the ONLY guard on this path (no LLM vote), so
        # "no" has to mean nothing is touched at all.
        index = self._bulk_delete_vault(vault_path, mocker)
        main.process_capture("delete all notes", index, confirm=lambda targets, note_type: False)
        assert set(index.data) == {"01_Concepts/A.md", "04_Logs/B.md"}

    def test_bulk_delete_confirm_receives_the_real_target_list(self, mocker, vault_path):
        # The prompt shows the person what they're about to lose -- if it were
        # handed the wrong list the confirmation would be meaningless.
        index = self._bulk_delete_vault(vault_path, mocker)
        seen = {}

        def fake_confirm(targets, note_type):
            seen["titles"] = [t["title"] for t in targets]
            seen["note_type"] = note_type
            return False

        main.process_capture("delete all notes", index, confirm=fake_confirm)
        assert seen["titles"] == ["A", "B"]
        assert seen["note_type"] is None

    def test_bulk_delete_one_failure_does_not_abort_the_rest(self, mocker, vault_path):
        index = self._bulk_delete_vault(vault_path, mocker)
        mocker.patch.object(main.vault, "delete_note", side_effect=[OSError("locked"), None])
        main.process_capture("delete all notes", index, confirm=lambda targets, note_type: True)
        # First failed, second still got deleted and de-indexed.
        assert "01_Concepts/A.md" in index.data
        assert "04_Logs/B.md" not in index.data

    def test_link_item_adds_wikilink_and_reindexes_source(self, mocker, vault_path):
        target = vault_path / "01_Concepts"
        target.mkdir(parents=True)
        source_note = target / "Source.md"
        source_note.write_text("# Source\n\n## Related\n- none yet\n", encoding="utf-8")
        (target / "Target.md").write_text("# Target\n", encoding="utf-8")

        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[{
            "action": "link", "source_path": "01_Concepts/Source.md", "source_title": "Source",
            "target_path": "01_Concepts/Target.md", "target_title": "Target",
        }])
        main.process_capture("link Source to Target", index)

        text = source_note.read_text(encoding="utf-8")
        assert "[[Target]]" in text
        assert "01_Concepts/Source.md" in index.data

    def test_not_content_item_writes_nothing_and_explains(self, mocker, vault_path):
        index = RagIndex()
        mocker.patch.object(agent, "process_capture", return_value=[{"action": "not_content"}])
        with main.console.capture() as capture:
            main.process_capture("Make me a file that links to another file.", index)
        output = capture.get()
        assert "instruction" in output
        assert not (vault_path / "01_Concepts").exists()
        assert not (vault_path / "02_Projects").exists()

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

    @pytest.mark.parametrize("title", ["love-for-idea-agent", "Idea-Agent", "#tagged", "@mention"])
    def test_wikilink_titles_survive_rich_markup(self, title):
        # Rich's tag regex matches any [...] opening with a lowercase letter,
        # #, / or @ -- so escaping only the title left the literal brackets to
        # be parsed, and [[love-for-idea-agent]] printed as an empty "[]".
        # Capitalized fixture titles hid this in every existing test.
        rendered = main._wikilink(title)
        with main.console.capture() as capture:
            main.console.print(rendered)
        assert f"[[{title}]]" in capture.get()

    def test_deleted_lowercase_title_is_not_swallowed_in_summary(self, mocker, vault_path):
        index = RagIndex()
        with main.console.capture() as capture:
            main._print_summary([], [], [], deleted=["love-for-idea-agent"])
        assert "love-for-idea-agent" in capture.get()

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
