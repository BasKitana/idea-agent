import json

import pytest

import agent


def chat_returning(content):
    if not isinstance(content, str):
        content = json.dumps(content)
    return {"message": {"content": content}}


RELATED = [{"title": "Existing Note", "path": "Health/Existing Note.md", "score": 0.9}]


class TestWellFormed:
    def test_valid_response_passes_through(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning({
            "title": "New Idea", "folder": "Health", "tags": ["a", "b"],
            "body": "summary", "related_titles": ["Existing Note"],
        }))
        result = agent.classify_idea("idea text", ["Health"], RELATED)
        assert result == {
            "title": "New Idea", "folder": "Health", "tags": ["a", "b"],
            "body": "summary", "related_titles": ["Existing Note"],
        }

    def test_extra_unknown_keys_are_ignored(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning({
            "title": "X", "folder": "Health", "tags": [], "body": "b",
            "related_titles": [], "confidence": 0.9,
        }))
        result = agent.classify_idea("idea", [], [])
        assert result["title"] == "X"

    def test_case_insensitive_keys(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning({
            "Title": "X", "Folder": "Health", "Tags": ["a"], "Body": "b", "Related_Titles": [],
        }))
        result = agent.classify_idea("idea", [], [])
        assert result["title"] == "X"
        assert result["folder"] == "Health"


class TestMissingOrNullFields:
    @pytest.mark.parametrize("field", ["title", "folder", "tags", "body", "related_titles"])
    def test_missing_field_still_returns_other_valid_fields(self, mocker, field):
        payload = {"title": "X", "folder": "Health", "tags": ["a"], "body": "b", "related_titles": []}
        del payload[field]
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(payload))
        result = agent.classify_idea("fallback idea text", [], [])
        # Every other field must survive even though one was missing.
        for k, v in payload.items():
            assert result[k] == v

    @pytest.mark.parametrize("field", ["tags", "related_titles"])
    def test_explicit_null_defaults_to_empty_list(self, mocker, field):
        payload = {"title": "X", "folder": "Health", "tags": ["a"], "body": "b", "related_titles": []}
        payload[field] = None
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(payload))
        result = agent.classify_idea("idea", [], [])
        assert result[field] == []

    def test_null_title_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": None, "folder": "Health", "tags": [], "body": "b", "related_titles": []}))
        result = agent.classify_idea("short idea", [], [])
        assert result["title"] == "short idea"

    def test_empty_string_title_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "", "folder": "Health", "tags": [], "body": "b", "related_titles": []}))
        result = agent.classify_idea("short idea", [], [])
        assert result["title"] == "short idea"

    def test_whitespace_only_title_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "   ", "folder": "Health", "tags": [], "body": "b", "related_titles": []}))
        result = agent.classify_idea("short idea", [], [])
        assert result["title"] == "short idea"

    def test_whitespace_only_folder_falls_back_to_inbox(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "X", "folder": "   ", "tags": [], "body": "b", "related_titles": []}))
        result = agent.classify_idea("idea", [], [])
        assert result["folder"] == "Inbox"

    def test_whitespace_only_body_falls_back_to_idea_text(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "X", "folder": "Health", "tags": [], "body": " ", "related_titles": []}))
        result = agent.classify_idea("the original idea", [], [])
        assert result["body"] == "the original idea"


class TestWrongTypes:
    def test_tags_as_bare_string_becomes_single_item_list(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "X", "folder": "Health", "tags": "brainstorm", "body": "b", "related_titles": []}))
        result = agent.classify_idea("idea", [], [])
        assert result["tags"] == ["brainstorm"]

    def test_related_titles_as_bare_string_becomes_single_item_list(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "X", "folder": "Health", "tags": [], "body": "b", "related_titles": "Existing Note"}))
        result = agent.classify_idea("idea", [], RELATED)
        assert result["related_titles"] == ["Existing Note"]

    def test_tags_list_with_null_and_nested_object_are_dropped_or_stringified(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "X", "folder": "Health", "tags": [None, "ok", 42], "body": "b", "related_titles": []}))
        result = agent.classify_idea("idea", [], [])
        assert "ok" in result["tags"] and "42" in result["tags"]
        assert None not in result["tags"] and "None" not in result["tags"]


class TestHallucinationGuard:
    def test_related_title_not_in_supplied_list_is_dropped(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "X", "folder": "Health", "tags": [], "body": "b",
             "related_titles": ["Made Up Note That Does Not Exist"]}))
        result = agent.classify_idea("idea", [], RELATED)
        assert result["related_titles"] == []

    def test_genuine_related_title_survives(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(
            {"title": "X", "folder": "Health", "tags": [], "body": "b",
             "related_titles": ["Existing Note"]}))
        result = agent.classify_idea("idea", [], RELATED)
        assert result["related_titles"] == ["Existing Note"]


class TestNonDictOrInvalidJson:
    @pytest.mark.parametrize("content", ['["a","b"]', '"hello"', "42", "null", "true"])
    def test_non_dict_json_falls_back_to_inbox(self, mocker, content):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(content))
        result = agent.classify_idea("my idea", [], [])
        assert result["folder"] == "Inbox"

    @pytest.mark.parametrize("content", [
        "", "not json at all", '{"title":"x",}', "{'title': 'x'}", '{"title": "Foo", "folder": "Busi',
    ])
    def test_invalid_json_falls_back_to_inbox(self, mocker, content):
        mocker.patch.object(agent.ollama, "chat", return_value=chat_returning(content))
        result = agent.classify_idea("my idea", [], [])
        assert result["folder"] == "Inbox"

    def test_missing_message_key_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value={})
        result = agent.classify_idea("my idea", [], [])
        assert result["folder"] == "Inbox"

    def test_none_content_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", return_value={"message": {"content": None}})
        result = agent.classify_idea("my idea", [], [])
        assert result["folder"] == "Inbox"


class TestConnectionFailures:
    def test_connection_error_then_success_recovers(self, mocker):
        good = chat_returning({"title": "X", "folder": "Health", "tags": [], "body": "b", "related_titles": []})
        mocker.patch.object(agent.ollama, "chat", side_effect=[ConnectionError("down"), good])
        result = agent.classify_idea("idea", [], [])
        assert result["title"] == "X"

    def test_connection_error_on_both_calls_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=ConnectionError("down"))
        result = agent.classify_idea("my idea", [], [])
        assert result == {
            "title": "my idea", "folder": "Inbox", "tags": ["unsorted"],
            "body": "my idea", "related_titles": [],
        }

    def test_generic_exception_on_both_calls_falls_back(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=Exception("boom"))
        result = agent.classify_idea("my idea", [], [])
        assert result["folder"] == "Inbox"


class TestInboxFallbackTitle:
    def test_short_idea_used_verbatim(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=Exception("boom"))
        result = agent.classify_idea("short", [], [])
        assert result["title"] == "short"

    def test_empty_idea_falls_back_to_untitled(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=Exception("boom"))
        result = agent.classify_idea("", [], [])
        assert result["title"] == "Untitled Idea"

    def test_whitespace_only_idea_falls_back_to_untitled(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=Exception("boom"))
        result = agent.classify_idea("   \n\t", [], [])
        assert result["title"] == "Untitled Idea"

    def test_long_idea_truncated_to_sixty_chars(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=Exception("boom"))
        result = agent.classify_idea("x" * 100, [], [])
        assert len(result["title"]) == 60
