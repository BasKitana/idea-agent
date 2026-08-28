import json

import pytest

import agent


def chat_returning(content):
    if not isinstance(content, str):
        content = json.dumps(content)
    return {"message": {"content": content}}


CANDIDATES = [
    {"title": "Existing Note", "type": "project", "path": "02_Projects/Existing Note.md",
     "score": 0.6, "excerpt": "Something related."},
]


def dup_response(title):
    return chat_returning({"duplicate_of": title})


def no_dup_response():
    return chat_returning({"duplicate_of": None})


def atomize_response(notes):
    return chat_returning({"notes": notes})


class TestDuplicateDetection:
    def test_no_candidates_skips_duplicate_check_entirely(self, mocker):
        spy = mocker.patch.object(agent.ollama, "chat", return_value=atomize_response(
            [{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]))
        agent.process_capture("idea with no candidates", [])
        # Only the atomize call should have happened -- no wasted duplicate-check call.
        assert spy.call_count == 1

    def test_first_vote_duplicate_short_circuits_atomize(self, mocker):
        spy = mocker.patch.object(agent.ollama, "chat", return_value=dup_response("Existing Note"))
        result = agent.process_capture("idea", CANDIDATES)
        assert result == [{"action": "duplicate", "duplicate_of": "Existing Note", "note": ""}]
        assert spy.call_count == 1  # atomize never called

    def test_or_ensemble_catches_a_late_positive_vote(self, mocker):
        # First two votes say "no", third says "yes" -- OR-ensemble must still catch it.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            no_dup_response(), no_dup_response(), dup_response("Existing Note"),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["action"] == "duplicate"
        assert result[0]["duplicate_of"] == "Existing Note"

    def test_all_votes_no_falls_through_to_atomize(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            no_dup_response(), no_dup_response(), no_dup_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["action"] == "create"
        assert result[0]["title"] == "New"

    def test_hallucinated_duplicate_target_is_ignored(self, mocker):
        # Model names a candidate that was never actually offered -- must not
        # be trusted, and must not crash; falls through to atomize.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            dup_response("Some Note That Was Never A Candidate"),
            dup_response("Also Not Real"),
            dup_response("Still Not Real"),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["action"] == "create"

    def test_duplicate_check_exception_treated_as_no_and_does_not_crash(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            Exception("boom"), Exception("boom"), Exception("boom"),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["action"] == "create"


class TestAtomize:
    def test_single_concept_produces_one_note(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=atomize_response(
            [{"title": "X", "type": "concept", "tags": ["a"], "body": "b", "links": []}]))
        result = agent.process_capture("idea", [])
        assert len(result) == 1
        assert result[0] == {"action": "create", "title": "X", "type": "concept",
                              "tags": ["a"], "body": "b", "links": []}

    def test_compound_capture_produces_multiple_linked_notes(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=atomize_response([
            {"title": "Project A", "type": "project", "tags": [], "body": "b1", "links": []},
            {"title": "Concept B", "type": "concept", "tags": [], "body": "b2", "links": ["Project A"]},
        ]))
        result = agent.process_capture("compound idea", [])
        assert len(result) == 2
        assert result[1]["links"] == ["Project A"]

    def test_link_to_retrieved_candidate_survives(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            no_dup_response(), no_dup_response(), no_dup_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [],
                                "body": "b", "links": ["Existing Note"]}]),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["links"] == ["Existing Note"]

    def test_multiple_notes_missing_titles_get_distinct_fallbacks(self, mocker):
        # Both notes omit "title" -- if they collapsed to the same fallback,
        # unique_path() would still avoid a file overwrite, but a link meant
        # for the second note would resolve (by filename) to the first.
        mocker.patch.object(agent.ollama, "chat", return_value=atomize_response([
            {"type": "concept", "tags": [], "body": "b1", "links": []},
            {"type": "concept", "tags": [], "body": "b2", "links": []},
        ]))
        result = agent.process_capture("shared fallback capture", [])
        titles = [item["title"] for item in result]
        assert len(titles) == len(set(titles)), f"titles collided: {titles}"

    def test_hallucinated_link_is_dropped(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=atomize_response(
            [{"title": "X", "type": "concept", "tags": [], "body": "b",
              "links": ["Nonexistent Note"]}]))
        result = agent.process_capture("idea", [])
        assert result[0]["links"] == []

    @pytest.mark.parametrize("bad_type", ["invalid", "Idea", "", None, 42])
    def test_invalid_type_coerces_to_concept(self, mocker, bad_type):
        mocker.patch.object(agent.ollama, "chat", return_value=atomize_response(
            [{"title": "X", "type": bad_type, "tags": [], "body": "b", "links": []}]))
        result = agent.process_capture("idea", [])
        assert result[0]["type"] == "concept"

    def test_tags_as_bare_string_becomes_single_item_list(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=atomize_response(
            [{"title": "X", "type": "concept", "tags": "brainstorm", "body": "b", "links": []}]))
        result = agent.process_capture("idea", [])
        assert result[0]["tags"] == ["brainstorm"]

    def test_missing_title_falls_back_to_capture_text(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=atomize_response(
            [{"type": "concept", "tags": [], "body": "b", "links": []}]))
        result = agent.process_capture("fallback capture text", [])
        assert result[0]["title"] == "fallback capture text"

    def test_empty_notes_list_retries_then_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            atomize_response([]), atomize_response([]),
        ])
        result = agent.process_capture("my capture", [])
        assert result == [{"action": "create", "title": "my capture", "type": "concept",
                            "tags": ["unsorted"], "body": "my capture", "links": []}]

    @pytest.mark.parametrize("content", [
        "not json", '{"notes":', "", "null", '{"notes": "not a list"}',
    ])
    def test_malformed_atomize_response_falls_back(self, mocker, content):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(content))
        result = agent.process_capture("my capture", [])
        assert result[0]["action"] == "create"
        assert result[0]["type"] == "concept"

    def test_connection_error_then_success_recovers(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            ConnectionError("down"),
            atomize_response([{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", [])
        assert result[0]["title"] == "X"

    def test_connection_error_on_both_attempts_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=ConnectionError("down"))
        result = agent.process_capture("my capture", [])
        assert result == [{"action": "create", "title": "my capture", "type": "concept",
                            "tags": ["unsorted"], "body": "my capture", "links": []}]

    def test_empty_capture_falls_back_to_untitled(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=Exception("boom"))
        result = agent.process_capture("", [])
        assert result[0]["title"] == "Untitled"

    def test_long_capture_truncated_to_sixty_chars_in_fallback(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=Exception("boom"))
        result = agent.process_capture("x" * 100, [])
        assert len(result[0]["title"]) == 60
