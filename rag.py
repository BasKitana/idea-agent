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
        if config.INDEX_PATH.exists():
            return json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))
        return {}

    def save(self):
        config.INDEX_PATH.write_text(json.dumps(self.data), encoding="utf-8")

    def sync(self):
        for key in list(self.data.keys()):
            if not (config.VAULT_PATH / key).exists():
                del self.data[key]

        for md_path in config.VAULT_PATH.rglob("*.md"):
            rel = md_path.relative_to(config.VAULT_PATH)
            if any(part.startswith(".") for part in rel.parts):
                continue
            key = str(rel)
            mtime = md_path.stat().st_mtime
            entry = self.data.get(key)
            if entry and entry["mtime"] == mtime:
                continue
            text = md_path.read_text(encoding="utf-8")
            title = md_path.stem
            embedding = self.model.encode(text).tolist()
            self.data[key] = {"mtime": mtime, "title": title, "embedding": embedding}
        self.save()

    def add(self, path: Path, title: str, text: str):
        key = str(path.relative_to(config.VAULT_PATH))
        embedding = self.model.encode(text).tolist()
        self.data[key] = {"mtime": path.stat().st_mtime, "title": title, "embedding": embedding}
        self.save()

    def query(self, text: str, top_k: int = None, min_score: float = 0.35) -> list[dict]:
        if not self.data:
            return []
        top_k = top_k or config.TOP_K
        query_vec = np.array(self.model.encode(text))
        query_vec = query_vec / np.linalg.norm(query_vec)

        scored = []
        for key, entry in self.data.items():
            vec = np.array(entry["embedding"])
            vec = vec / np.linalg.norm(vec)
            score = float(np.dot(query_vec, vec))
            if score >= min_score:
                scored.append({"title": entry["title"], "path": key, "score": score})

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]
