import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer: same text -> same vector,
    no model download, no GPU/CPU inference cost."""

    def encode(self, text):
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.rand(8)


@pytest.fixture
def vault_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VAULT_PATH", tmp_path)
    monkeypatch.setattr(config, "INDEX_PATH", tmp_path / ".rag_index.json")
    return tmp_path


@pytest.fixture
def rag_index(vault_path):
    import rag

    index = rag.RagIndex()
    index._model = FakeEmbedder()
    return index
