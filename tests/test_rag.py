import json

import numpy as np
import pytest

import config
import rag


def make_note(vault_path, folder, name, text="hello world"):
    d = vault_path / folder
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.md"
    p.write_text(text, encoding="utf-8")
    return p


class TestLoad:
    def test_missing_index_file_returns_empty(self, vault_path):
        assert rag.RagIndex().data == {}

    @pytest.mark.parametrize("content", ["", "   ", "{not json", "[1, 2, 3]", '"a string"', "42"])
    def test_corrupt_or_wrong_shape_index_self_heals_to_empty(self, vault_path, content):
        config.INDEX_PATH.write_text(content, encoding="utf-8")
        assert rag.RagIndex().data == {}

    def test_valid_dict_index_loads(self, vault_path):
        config.INDEX_PATH.write_text(json.dumps({"a.md": {"mtime": 1, "title": "A", "embedding": [1, 0]}}))
        assert "a.md" in rag.RagIndex().data


class TestSync:
    def test_empty_vault_produces_empty_index(self, rag_index):
        rag_index.sync()
        assert rag_index.data == {}

    def test_missing_vault_path_does_not_wipe_existing_index(self, vault_path, rag_index, monkeypatch):
        rag_index.data = {"a.md": {"mtime": 1, "title": "A", "embedding": [1, 0]}}
        rag_index.save()
        monkeypatch.setattr(config, "VAULT_PATH", vault_path / "does-not-exist")
        rag_index.sync()
        assert "a.md" in rag_index.data, "sync() must not wipe the index when VAULT_PATH is unreachable"

    def test_new_note_gets_indexed(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Water")
        rag_index.sync()
        assert "Health/Water.md" in rag_index.data

    def test_deleted_note_gets_pruned(self, vault_path, rag_index):
        p = make_note(vault_path, "Health", "Water")
        rag_index.sync()
        p.unlink()
        rag_index.sync()
        assert rag_index.data == {}

    def test_dot_prefixed_folder_is_skipped(self, vault_path, rag_index):
        make_note(vault_path, ".obsidian", "plugin-data")
        rag_index.sync()
        assert rag_index.data == {}

    def test_hidden_file_at_root_is_skipped(self, vault_path, rag_index):
        (vault_path / ".scratch.md").write_text("secret", encoding="utf-8")
        rag_index.sync()
        assert rag_index.data == {}

    def test_unreadable_file_does_not_abort_the_whole_pass(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Good")
        bad = vault_path / "Health" / "Bad.md"
        bad.write_bytes(b"\xff\xfe\x00\xff not valid utf-8 alone")
        rag_index.sync()
        assert "Health/Good.md" in rag_index.data

    def test_unchanged_mtime_skips_reembedding(self, vault_path, rag_index, monkeypatch):
        make_note(vault_path, "Health", "Water")
        rag_index.sync()
        calls = []
        monkeypatch.setattr(rag_index._model, "encode", lambda t: calls.append(t) or rag_index._model.encode(t))
        rag_index.sync()
        assert calls == []

    def test_path_separator_normalized_to_posix(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Water")
        rag_index.sync()
        assert all("\\" not in key for key in rag_index.data)


class TestQuery:
    def test_empty_index_returns_empty(self, rag_index):
        assert rag_index.query("anything") == []

    def test_top_k_zero_returns_nothing(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Water")
        rag_index.sync()
        assert rag_index.query("water", top_k=0) == []

    def test_top_k_none_uses_config_default(self, vault_path, rag_index):
        for i in range(10):
            make_note(vault_path, "Health", f"Note{i}", text=f"water idea number {i}")
        rag_index.sync()
        assert len(rag_index.query("water idea", top_k=None, min_score=-1)) == config.TOP_K

    def test_top_k_larger_than_index_is_clamped(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Water")
        rag_index.sync()
        assert len(rag_index.query("water", top_k=1000, min_score=-1)) == 1

    def test_identical_text_scores_near_one(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Water", text="daily water intake tracker")
        rag_index.sync()
        results = rag_index.query("daily water intake tracker", min_score=-1)
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)

    def test_all_below_min_score_returns_empty(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Water")
        rag_index.sync()
        assert rag_index.query("water", min_score=1.1) == []

    def test_malformed_entry_is_skipped_not_fatal(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Good")
        rag_index.sync()
        rag_index.data["Health/Bad.md"] = {"mtime": 1, "title": "Bad", "embedding": "not-a-vector"}
        results = rag_index.query("good", min_score=-1)
        titles = [r["title"] for r in results]
        assert "Good" in titles

    def test_dimension_mismatched_embedding_is_skipped_not_fatal(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Good")
        rag_index.sync()
        rag_index.data["Health/Old.md"] = {"mtime": 1, "title": "Old", "embedding": [1, 2, 3, 4, 5]}
        results = rag_index.query("good", min_score=-1)
        titles = [r["title"] for r in results]
        assert "Good" in titles and "Old" not in titles

    def test_zero_vector_entry_is_skipped_not_nan(self, vault_path, rag_index):
        make_note(vault_path, "Health", "Good")
        rag_index.sync()
        dim = len(next(iter(rag_index.data.values()))["embedding"])
        rag_index.data["Health/Zero.md"] = {"mtime": 1, "title": "Zero", "embedding": [0] * dim}
        results = rag_index.query("good", min_score=-1)
        assert all(not np.isnan(r["score"]) for r in results)
