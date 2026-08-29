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


def not_instruction_response():
    return chat_returning({"is_instruction": False})


def instruction_response():
    return chat_returning({"is_instruction": True})


def delete_command_response(target):
    return chat_returning({"action": "delete", "target": target})


def link_command_response(source, target):
    return chat_returning({"action": "link", "source": source, "target": target})


def unclear_command_response():
    return chat_returning({"action": "unclear"})


def confirm_response(confirmed):
    return chat_returning({"confirmed": confirmed})


KNOWN_NOTES = [
    {"title": "Old Note", "path": "01_Concepts/Old Note.md"},
    {"title": "Other Note", "path": "01_Concepts/Other Note.md"},
]


def no_append_response():
    return chat_returning({"append_to": None})


def append_response(title):
    return chat_returning({"append_to": title})


def redundant_response():
    return chat_returning({"redundant": True})


def no_redundant_response():
    return chat_returning({"redundant": False})


# process_capture() also runs a redundant-update check (up to 3 UNANIMOUS
# votes) whenever the append-check actually wins -- i.e. any test where an
# "append" action is the final expected result. all() short-circuits on the
# first falsy vote, so a single no_redundant_response() is enough to make the
# append win; only a test that specifically wants the redundant-suppression
# path to fire needs 3 redundant_response() items.


# process_capture() also runs an append-check (up to 3 OR-ensemble votes,
# same shape as duplicate-check) whenever candidates contain exactly one
# entry at/above AUTO_LINK_SCORE -- i.e. every test using CANDIDATES (score
# 0.6) or a locally-defined single strong candidate. Any side_effect list
# for those must account for it (3 more items between the duplicate votes
# and the atomize response) or the sequence silently degrades to the
# fallback-on-exhaustion path instead of testing what it says it tests.


# process_capture() now always runs the meta-command check first. `return_value`
# mocks (a single response reused for every call) are naturally safe -- none of
# dup_response/no_dup_response/atomize_response include an "is_instruction" key,
# so the meta-check reads it as absent/False and proceeds normally. Any test
# using a `side_effect` LIST must account for this extra leading call, or the
# whole sequence shifts by one and later assertions fail for the wrong reason.


class TestSessionHistory:
    """Short-term memory: recent {"capture","summary"} turns from this REPL
    session, so a pronoun-only capture like "delete it" can resolve what
    "it" refers to -- RAG retrieval alone can't, since pronouns carry almost
    no embedding signal."""

    def test_empty_history_produces_empty_block(self):
        assert agent._format_session_history([]) == ""
        assert agent._format_session_history(None) == ""

    def test_history_entries_appear_in_formatted_block(self):
        history = [
            {"capture": "Idea Agent uses Ollama", "summary": "filed 'Idea Agent' (project)"},
            {"capture": "it also has RAG", "summary": "updated 'Idea Agent'"},
        ]
        block = agent._format_session_history(history)
        assert "Idea Agent uses Ollama" in block
        assert "filed 'Idea Agent' (project)" in block
        assert "it also has RAG" in block

    def test_session_history_reaches_the_meta_check_prompt(self, mocker):
        # Proves the plumbing actually wires through process_capture() into
        # the real LLM call, not just that the formatter works in isolation.
        spy = mocker.patch.object(agent.ollama, "chat", return_value=instruction_response())
        history = [{"capture": "Idea Agent Project", "summary": "filed 'Idea Agent Project' (project)"}]
        agent.process_capture("delete it", [], [{"title": "Idea Agent Project", "path": "x.md"}], history)
        sent_prompt = spy.call_args_list[0].kwargs["messages"][1]["content"]
        assert "Idea Agent Project" in sent_prompt

    def test_session_history_reaches_the_atomize_prompt(self, mocker):
        spy = mocker.patch.object(agent.ollama, "chat", return_value=atomize_response(
            [{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]))
        history = [{"capture": "a prior idea", "summary": "filed 'X' (concept)"}]
        agent.process_capture("a completely different fresh idea with enough words", [], [], history)
        # Last call in the sequence is the atomize call (meta-check short-circuits
        # to False immediately since the mocked response lacks "is_instruction").
        sent_prompt = spy.call_args_list[-1].kwargs["messages"][1]["content"]
        assert "a prior idea" in sent_prompt


class TestMetaCommandDetection:
    def test_instruction_short_circuits_before_any_other_call(self, mocker):
        # return_value repeats {"is_instruction": True} for every call,
        # including the command-parse call -- which then has no "action" key,
        # so it correctly falls through to "unclear" -> not_content.
        spy = mocker.patch.object(agent.ollama, "chat", return_value=instruction_response())
        result = agent.process_capture("Make me a file that links to another file.", CANDIDATES)
        assert result == [{"action": "not_content"}]
        # Unanimous vote: confirming all 3 agree costs 3 calls (can't short-circuit
        # on agreement, only on the first disagreement), plus 1 command-parse call --
        # but still short-circuits before ever reaching duplicate-check or atomize.
        assert spy.call_count == 4

    def test_non_instruction_proceeds_to_normal_pipeline(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            atomize_response([{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("a real idea", [])
        assert result[0]["action"] == "create"
        assert result[0]["title"] == "X"

    def test_meta_check_exception_fails_open(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            Exception("boom"),
            atomize_response([{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("a real idea", [])
        assert result[0]["action"] == "create"

    def test_malformed_meta_check_response_fails_open(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            chat_returning("not json"),
            atomize_response([{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("a real idea", [])
        assert result[0]["action"] == "create"

    def test_missing_is_instruction_key_defaults_to_false(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            chat_returning({}),
            atomize_response([{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("a real idea", [])
        assert result[0]["action"] == "create"


class TestDuplicateDetection:
    def test_no_candidates_skips_duplicate_check_entirely(self, mocker):
        spy = mocker.patch.object(agent.ollama, "chat", return_value=atomize_response(
            [{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]))
        agent.process_capture("idea with no candidates", [])
        # Meta-check (1) + atomize (1) -- duplicate-check adds nothing since
        # there are no candidates to compare against.
        assert spy.call_count == 2

    def test_first_vote_duplicate_short_circuits_atomize(self, mocker):
        spy = mocker.patch.object(agent.ollama, "chat", return_value=dup_response("Existing Note"))
        result = agent.process_capture("idea", CANDIDATES)
        assert result == [{"action": "duplicate", "duplicate_of": "Existing Note", "note": ""}]
        assert spy.call_count == 2  # meta-check (1) + first duplicate vote (1); atomize never called

    def test_or_ensemble_catches_a_late_positive_vote(self, mocker):
        # First two votes say "no", third says "yes" -- OR-ensemble must still catch it.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), dup_response("Existing Note"),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["action"] == "duplicate"
        assert result[0]["duplicate_of"] == "Existing Note"

    def test_all_votes_no_falls_through_to_atomize(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            no_append_response(), no_append_response(), no_append_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["action"] == "create"
        assert result[0]["title"] == "New"

    def test_hallucinated_duplicate_target_is_ignored(self, mocker):
        # Model names a candidate that was never actually offered -- must not
        # be trusted, and must not crash; falls through to atomize.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            dup_response("Some Note That Was Never A Candidate"),
            dup_response("Also Not Real"),
            dup_response("Still Not Real"),
            no_append_response(), no_append_response(), no_append_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["action"] == "create"

    def test_duplicate_check_exception_treated_as_no_and_does_not_crash(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            Exception("boom"), Exception("boom"), Exception("boom"),
            no_append_response(), no_append_response(), no_append_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", CANDIDATES)
        assert result[0]["action"] == "create"


class TestRelationHint:
    """Pure-function tests for the prompt hint that names a dominant related
    candidate's type explicitly. Measured directly: without this, type
    classification for a capture strongly tied to an existing project
    scattered across concept/project/log (5/8, 1/8, 2/8) even though the
    related note was already in the candidate list."""

    def test_single_strong_candidate_produces_a_hint(self):
        candidates = [{"title": "Idea Agent Project", "type": "project", "score": 0.7}]
        hint = agent._relation_hint(candidates)
        assert "Idea Agent Project" in hint and "project" in hint

    def test_no_strong_candidates_produces_no_hint(self):
        candidates = [{"title": "Weak Match", "type": "concept", "score": 0.1}]
        assert agent._relation_hint(candidates) == ""

    def test_multiple_strong_candidates_produces_no_hint(self):
        # Ambiguous which one the capture is actually about -- don't guess.
        candidates = [
            {"title": "A", "type": "project", "score": 0.7},
            {"title": "B", "type": "concept", "score": 0.6},
        ]
        assert agent._relation_hint(candidates) == ""

    def test_empty_candidates_produces_no_hint(self):
        assert agent._relation_hint([]) == ""


class TestAutoLink:
    """Measured directly: given a strongly-related candidate right there in
    the prompt, the model only populated "links" in 4/6 runs -- it forgets,
    the same unreliability pattern as duplicate-detection. Since the
    retrieval score is already known before the LLM runs, auto-link anything
    at/above AUTO_LINK_SCORE regardless of what the LLM decided."""

    def test_high_score_candidate_gets_linked_even_if_llm_omits_it(self, mocker):
        candidates = [{"title": "Existing Note", "type": "project", "score": 0.90, "excerpt": "x"}]
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            no_append_response(), no_append_response(), no_append_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", candidates)
        assert "Existing Note" in result[0]["links"]

    def test_low_score_candidate_is_not_auto_linked(self, mocker):
        candidates = [{"title": "Existing Note", "type": "project", "score": 0.10, "excerpt": "x"}]
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", candidates)
        assert result[0]["links"] == []

    def test_auto_link_merges_without_duplicating_llms_own_link(self, mocker):
        candidates = [{"title": "Existing Note", "type": "project", "score": 0.90, "excerpt": "x"}]
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            no_append_response(), no_append_response(), no_append_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b",
                                "links": ["Existing Note"]}]),
        ])
        result = agent.process_capture("idea", candidates)
        assert result[0]["links"].count("Existing Note") == 1

    def test_auto_link_scoped_to_single_note_batches(self, mocker):
        # A multi-note split doesn't cleanly tell us which resulting note a
        # whole-capture-level retrieval score actually belongs to, so it's
        # scoped off rather than linking every sibling note indiscriminately.
        candidates = [{"title": "Existing Note", "type": "project", "score": 0.90, "excerpt": "x"}]
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            no_append_response(), no_append_response(), no_append_response(),
            atomize_response([
                {"title": "Note A", "type": "project", "tags": [], "body": "b1", "links": []},
                {"title": "Note B", "type": "concept", "tags": [], "body": "b2", "links": []},
            ]),
        ])
        result = agent.process_capture("compound idea", candidates)
        assert all("Existing Note" not in item["links"] for item in result)


class TestVaultCommands:
    """Delete/link execution for unambiguous, named-target vault
    instructions. Delete carries its own independent unanimous
    re-confirmation on top of the initial parse -- the one operation here
    with no recovery path but the OS recycle bin, so it's deliberately
    biased toward refusing over guessing."""

    def test_delete_with_pronoun_resolved_by_session_history(self, mocker):
        # Reported live: "delete it" resolved correctly to the right note at
        # parse time (which does see session_history), but confirmation
        # without that same context had no way to verify the resolution and
        # refused every time. Both stages need it.
        history = [{"capture": "Idea Agent Project", "summary": "filed 'Idea Agent Project' (project)"}]
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            delete_command_response("Old Note"),
            confirm_response(True), confirm_response(True), confirm_response(True),
        ])
        result = agent.process_capture("delete it", [], KNOWN_NOTES, history)
        assert result == [{
            "action": "delete", "target_path": "01_Concepts/Old Note.md", "target_title": "Old Note",
        }]
        # The confirm call must have actually received the history, not just parse.
        confirm_call = agent.ollama.chat.call_args_list[4]
        assert "Idea Agent Project" in confirm_call.kwargs["messages"][1]["content"]

    def test_delete_matches_title_despite_hyphen_case_difference(self, mocker):
        # Reported live: a note auto-titled "HNSW-indexing" by atomize
        # couldn't be found by an instruction naming it "HNSW indexing" --
        # same words, different incidental punctuation/case.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            delete_command_response("HNSW indexing"),
            confirm_response(True), confirm_response(True), confirm_response(True),
        ])
        notes = [{"title": "HNSW-indexing", "path": "01_Concepts/HNSW-indexing.md"}]
        result = agent.process_capture("delete HNSW indexing", [], notes)
        assert result == [{
            "action": "delete", "target_path": "01_Concepts/HNSW-indexing.md",
            "target_title": "HNSW-indexing",
        }]

    def test_delete_confirmed_executes(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            delete_command_response("Old Note"),
            confirm_response(True), confirm_response(True), confirm_response(True),
        ])
        result = agent.process_capture("delete Old Note", [], KNOWN_NOTES)
        assert result == [{
            "action": "delete", "target_path": "01_Concepts/Old Note.md", "target_title": "Old Note",
        }]

    def test_delete_target_not_a_real_note_is_refused(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            delete_command_response("Nonexistent Note"),
        ])
        result = agent.process_capture("delete Nonexistent Note", [], KNOWN_NOTES)
        assert result == [{"action": "not_content"}]

    def test_delete_confirmation_disagreement_is_refused(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            delete_command_response("Old Note"),
            confirm_response(True), confirm_response(False),
        ])
        result = agent.process_capture("delete Old Note", [], KNOWN_NOTES)
        assert result == [{"action": "not_content"}]

    def test_link_both_found_executes(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            link_command_response("Old Note", "Other Note"),
        ])
        result = agent.process_capture("link Old Note to Other Note", [], KNOWN_NOTES)
        assert result == [{
            "action": "link", "source_path": "01_Concepts/Old Note.md", "source_title": "Old Note",
            "target_path": "01_Concepts/Other Note.md", "target_title": "Other Note",
        }]

    def test_link_to_itself_is_refused(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            link_command_response("Old Note", "Old Note"),
        ])
        result = agent.process_capture("link Old Note to itself", [], KNOWN_NOTES)
        assert result == [{"action": "not_content"}]

    def test_link_with_hallucinated_target_is_refused(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            link_command_response("Old Note", "Ghost Note"),
        ])
        result = agent.process_capture("link Old Note to Ghost Note", [], KNOWN_NOTES)
        assert result == [{"action": "not_content"}]

    def test_unclear_action_is_refused(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            unclear_command_response(),
        ])
        result = agent.process_capture("clean up my vault", [], KNOWN_NOTES)
        assert result == [{"action": "not_content"}]

    def test_command_parse_exception_is_refused_not_crashed(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            Exception("boom"),
        ])
        result = agent.process_capture("delete Old Note", [], KNOWN_NOTES)
        assert result == [{"action": "not_content"}]

    def test_no_known_notes_defaults_to_empty_list(self, mocker):
        # process_capture() with known_notes omitted must not crash.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            instruction_response(), instruction_response(), instruction_response(),
            unclear_command_response(),
        ])
        result = agent.process_capture("delete something", [])
        assert result == [{"action": "not_content"}]


class TestAppendToExisting:
    """Reported directly: three near-duplicate notes about the same project
    ("Idea Agent Project", "...- Smart Feature", "...(2)") accumulated from
    captures that were really just small new facts about the same subject.
    A capture about an existing note's subject should merge into it instead
    of spawning another file.

    Mirrors TestDuplicateDetection's shape on purpose: _check_append now has
    the exact same architecture as _check_duplicate (every candidate offered
    to one judgment call, no score pre-filter) after the old single-
    dominant-candidate gate was found, via real usage, to skip the append
    judgment entirely whenever 0 or 2+ candidates crossed AUTO_LINK_SCORE --
    which is common, not rare, given how much embedding scores for related-
    but-differently-worded text overlap."""

    STRONG = [{"title": "Existing Note", "type": "project",
               "path": "02_Projects/Existing Note.md", "score": 0.90, "excerpt": "x"}]

    def test_append_vote_returns_append_action_with_correct_target(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            append_response("Existing Note"),
            no_redundant_response(),
        ])
        result = agent.process_capture("Existing Note now does X too", self.STRONG)
        assert result == [{
            "action": "append", "target_path": "02_Projects/Existing Note.md",
            "target_title": "Existing Note", "text": "Existing Note now does X too",
        }]

    def test_no_candidates_skips_append_check_entirely(self, mocker):
        spy = mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            atomize_response([{"title": "X", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea with no candidates", [])
        assert result[0]["title"] == "X"
        assert spy.call_count == 2  # meta-check (1) + atomize (1) -- no candidates to offer

    def test_weak_scoring_candidate_is_still_offered_to_append_check(self, mocker):
        # Not score-gated any more, same as duplicate-check -- a candidate
        # under AUTO_LINK_SCORE still gets a real judgment call, it just
        # isn't auto-linked/type-hinted (that's a separate mechanism).
        weak = [{"title": "Weak", "type": "concept", "path": "x.md", "score": 0.1, "excerpt": "x"}]
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            append_response("Weak"),
            no_redundant_response(),
        ])
        result = agent.process_capture("idea", weak)
        assert result[0]["action"] == "append"
        assert result[0]["target_title"] == "Weak"

    def test_llm_picks_the_right_one_of_several_candidates(self, mocker):
        several = [
            {"title": "A", "type": "project", "path": "a.md", "score": 0.9, "excerpt": "x"},
            {"title": "B", "type": "concept", "path": "b.md", "score": 0.9, "excerpt": "x"},
        ]
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            append_response("B"),
            no_redundant_response(),
        ])
        result = agent.process_capture("idea", several)
        assert result[0]["action"] == "append"
        assert result[0]["target_title"] == "B"

    def test_hallucinated_append_target_is_ignored(self, mocker):
        # Model names a note that was never actually offered -- must not be
        # trusted, and must not crash; falls through to atomize.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            append_response("Never Offered"),
            append_response("Also Not Real"),
            append_response("Still Not Real"),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", self.STRONG)
        assert result[0]["action"] == "create"

    def test_all_append_votes_null_falls_through_to_atomize(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            no_append_response(), no_append_response(), no_append_response(),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", self.STRONG)
        assert result[0]["action"] == "create"

    def test_or_ensemble_catches_a_late_positive_append_vote(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            no_append_response(), no_append_response(), append_response("Existing Note"),
            no_redundant_response(),
        ])
        result = agent.process_capture("idea", self.STRONG)
        assert result[0]["action"] == "append"

    def test_redundant_update_is_treated_as_duplicate_not_appended(self, mocker):
        """Reported live: broadening append to merge substantial same-subject
        content also let a reworded restatement of an existing fact through
        as if it were new -- polluting the note with a repeated line. A
        dedicated, unanimous redundant-check catches it and downgrades to a
        duplicate outcome (log + link, no write) instead."""
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            append_response("Existing Note"),
            redundant_response(), redundant_response(), redundant_response(),
        ])
        result = agent.process_capture("Existing Note, reworded", self.STRONG)
        assert result == [{"action": "duplicate", "duplicate_of": "Existing Note", "note": ""}]

    def test_single_non_redundant_vote_is_enough_to_keep_the_append(self, mocker):
        # Unanimous means ALL votes must agree it's redundant to suppress --
        # a single dissenting vote (even after two "redundant" votes) must
        # let the append through, the opposite bias from duplicate-detection.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            append_response("Existing Note"),
            redundant_response(), redundant_response(), no_redundant_response(),
        ])
        result = agent.process_capture("Existing Note now does X too", self.STRONG)
        assert result[0]["action"] == "append"

    def test_redundant_check_exception_fails_open_to_append(self, mocker):
        # Same fail-open direction as the append-check itself: an error
        # judging redundancy must not silently drop real content.
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            append_response("Existing Note"),
            Exception("boom"),
        ])
        result = agent.process_capture("Existing Note now does X too", self.STRONG)
        assert result[0]["action"] == "append"

    def test_append_check_exception_treated_as_no_and_falls_through(self, mocker):
        mocker.patch.object(agent.ollama, "chat", side_effect=[
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            Exception("boom"), Exception("boom"), Exception("boom"),
            atomize_response([{"title": "New", "type": "concept", "tags": [], "body": "b", "links": []}]),
        ])
        result = agent.process_capture("idea", self.STRONG)
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
            not_instruction_response(),
            no_dup_response(), no_dup_response(), no_dup_response(),
            no_append_response(), no_append_response(), no_append_response(),
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
            not_instruction_response(),
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
            not_instruction_response(),
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
