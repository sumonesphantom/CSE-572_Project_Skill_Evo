"""Ollama chat client whose cache key covers the full request (model, messages AND options),
so sampled generations at different temperatures/seeds never collide."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import requests


class Client:
    def __init__(self, model: str, cache_path: str, host: str = "http://localhost:11434",
                 num_ctx: int = 8192, max_tokens: int = 400):
        self.model, self.host, self.num_ctx, self.max_tokens = model, host.rstrip("/"), num_ctx, max_tokens
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT, secs REAL)")
        self._lock = threading.Lock()
        self.calls = self.hits = 0

    def chat(self, messages, temperature: float = 0.0, seed: int = 0, max_tokens: int | None = None, stop=None):
        """Returns (text, seconds_of_generation). Cached answers report their original time."""
        opts = {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx,
                "num_predict": max_tokens or self.max_tokens}
        if stop:
            opts["stop"] = list(stop)
        key = hashlib.sha256(json.dumps({"m": self.model, "msgs": messages, "o": opts}, sort_keys=True).encode()).hexdigest()
        with self._lock:
            row = self._db.execute("SELECT v, secs FROM c WHERE k=?", (key,)).fetchone()
        if row:
            self.hits += 1
            return row[0], row[1]
        payload = {"model": self.model, "messages": messages, "stream": False, "options": opts}
        t0 = time.time()
        for attempt in range(3):
            try:
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
                r.raise_for_status()
                text = r.json()["message"]["content"]
                break
            except (requests.RequestException, KeyError) as e:
                if attempt == 2:
                    raise RuntimeError(f"Ollama call failed: {e}") from e
                time.sleep(2 * (attempt + 1))
        secs = time.time() - t0
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?,?)", (key, text, secs))
            self._db.commit()
        self.calls += 1
        return text, secs
