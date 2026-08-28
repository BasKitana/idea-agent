import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import config


class RagIndex:
    def __init__(self):
        self._model = None
        self.data = self._load()

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(config.EMBED_MODEL)
        return self._model

    def _load(self) -> dict:
        if not config.INDEX_PATH.exists():
            return {}
        try:
            data = json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt or unreadable index (truncated write, bad encoding, etc.) --
            # self-heal to an empty index rather than crashing the whole app.
            return {}
        return data if isinstance(data, dict) else {}

    def save(self):
        config.INDEX_PATH.write_text(json.dumps(self.data), encoding="utf-8")

    def sync(self):
        if not config.VAULT_PATH.exists():
            # Don't let a transiently-missing vault (unmounted drive, bad path)
            # look like "every note was deleted" and wipe the whole index.
            return

        for key in list(self.data.keys()):
            if not (config.VAULT_PATH / key).exists():
                del self.data[key]

        for md_path in config.VAULT_PATH.rglob("*.md"):
            rel = md_path.relative_to(config.VAULT_PATH)
            if any(part.startswith(".") for part in rel.parts):
                continue
            key = rel.as_posix()
            try:
                mtime = md_path.stat().st_mtime
                entry = self.data.get(key)
                if entry and entry.get("mtime") == mtime:
                    continue
                text = md_path.read_text(encoding="utf-8")
                embedding = self.model.encode(text).tolist()
            except Exception:
                # One unreadable/locked/non-UTF8 file shouldn't block indexing
                # everything else that changed.
                continue
            self.data[key] = {"mtime": mtime, "title": md_path.stem, "embedding": embedding}
        self.save()

    def add(self, path: Path, title: str, text: str):
        key = path.relative_to(config.VAULT_PATH).as_posix()
        embedding = self.model.encode(text).tolist()
        self.data[key] = {"mtime": path.stat().st_mtime, "title": title, "embedding": embedding}
        self.save()

    def query(self, text: str, top_k: int = None, min_score: float = 0.35) -> list[dict]:
        if not self.data:
            return []
        top_k = config.TOP_K if top_k is None else top_k
        if top_k <= 0:
            return []

        query_vec = np.array(self.model.encode(text), dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        scored = []
        for key, entry in self.data.items():
            try:
                vec = np.array(entry["embedding"], dtype=float)
                if vec.shape != query_vec.shape:
                    continue
                norm = np.linalg.norm(vec)
                if norm == 0:
                    continue
                score = float(np.dot(query_vec, vec / norm))
            except Exception:
                # Malformed/mismatched entry (e.g. index built with a different
                # EMBED_MODEL) -- skip it, don't fail the whole search.
                continue
            if score >= min_score:
                scored.append({"title": entry.get("title", key), "path": key, "score": score})

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]
