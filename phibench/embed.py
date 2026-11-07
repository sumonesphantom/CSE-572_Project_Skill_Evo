"""Local sentence embeddings through Ollama (default nomic-embed-text), cached on disk. Free/local only."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np
import requests


class Embedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434",
                 cache_path: str | Path | None = None, prefix: str = "clustering: ", batch: int = 64):
        self.model, self.base_url, self.prefix, self.batch = model, base_url.rstrip("/"), prefix, batch
        self._lock = threading.Lock()
        self._conn = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(cache_path), check_same_thread=False, timeout=60)
            self._conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT)")

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}|{self.prefix}|{text}".encode()).hexdigest()

    def _get(self, k):
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute("SELECT v FROM emb WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put_many(self, items):
        if self._conn is None:
            return
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", [(k, json.dumps(v)) for k, v in items])
            self._conn.commit()

    def _request(self, texts: list[str]) -> list[list[float]]:
        r = requests.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts}, timeout=600)
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-normalised embeddings, one row per text."""
        keys = [self._key(t) for t in texts]
        vecs = [self._get(k) for k in keys]
        todo = [i for i, v in enumerate(vecs) if v is None]
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s + self.batch]
            out = self._request([self.prefix + texts[i] for i in idx])
            for i, v in zip(idx, out):
                vecs[i] = v
            self._put_many([(keys[i], vecs[i]) for i in idx])
        m = np.asarray(vecs, dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
