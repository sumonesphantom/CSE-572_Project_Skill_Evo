"""Local chat clients with a persistent response cache. Free and local only: no API keys.

Backends
- ollama : Ollama's /api/chat (default, e.g. model "phi4-mini")
- openai : any OpenAI-compatible /v1/chat/completions server running locally,
           e.g. `mlx_lm.server` serving Phi-4-mini with a LoRA adapter.

The cache key covers everything that can change an answer: namespace (label or model),
backend, messages, max_tokens, num_ctx, temperature and sample index. Give each LoRA
adapter its own `label`, because the server reports the same base-model name for all of them.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

import requests


class Cache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, meta TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            row = self._conn.execute("SELECT v, meta FROM cache WHERE k=?", (key,)).fetchone()
        return (row[0], json.loads(row[1] or "{}")) if row else None

    def put(self, key: str, value: str, meta: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, value, json.dumps(meta)))
            self._conn.commit()


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class ChatClient:
    def __init__(self, model: str, backend: str = "ollama", base_url: str | None = None,
                 label: str | None = None, cache: Cache | None = None, num_ctx: int = 16384,
                 max_tokens: int = 512, timeout: int = 600, think: bool | None = None):
        if backend not in ("ollama", "openai"):
            raise ValueError(f"unknown backend {backend!r}")
        self.model, self.backend = model, backend
        self.base_url = (base_url or ("http://localhost:11434" if backend == "ollama" else "http://localhost:8080")).rstrip("/")
        self.label = label or model
        self.cache, self.num_ctx, self.max_tokens = cache, num_ctx, max_tokens
        self.timeout, self.think = timeout, think
        self.calls = self.cache_hits = 0
        self._lock = threading.Lock()

    def key(self, messages, temperature: float, sample: int) -> str:
        blob = {"ns": self.label, "b": self.backend, "msgs": messages, "n": self.max_tokens,
                "ctx": self.num_ctx, "t": temperature, "s": sample}
        return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()

    def chat(self, messages: list[dict], temperature: float = 0.0, sample: int = 0) -> tuple[str, dict]:
        """Returns (text, meta); meta = {seconds, in_tokens, out_tokens, cached}."""
        k = self.key(messages, temperature, sample)
        if self.cache is not None:
            hit = self.cache.get(k)
            if hit is not None:
                with self._lock:
                    self.cache_hits += 1
                return hit[0], {**hit[1], "cached": True}
        t0 = time.time()
        text, meta = self._post(messages, temperature, sample)
        text = strip_think(text)
        meta = {**meta, "seconds": round(time.time() - t0, 3)}
        with self._lock:
            self.calls += 1
        if self.cache is not None:
            self.cache.put(k, text, meta)
        return text, {**meta, "cached": False}

    def _post(self, messages, temperature, sample) -> tuple[str, dict]:
        if self.backend == "ollama":
            url = f"{self.base_url}/api/chat"
            payload = {"model": self.model, "messages": messages, "stream": False,
                       "options": {"temperature": temperature, "seed": sample,
                                   "num_ctx": self.num_ctx, "num_predict": self.max_tokens}}
            if self.think is not None:
                payload["think"] = self.think
        else:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {"model": self.model, "messages": messages, "temperature": temperature,
                       "max_tokens": self.max_tokens, "seed": sample}
        last = None
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                j = r.json()
                if self.backend == "ollama":
                    return j["message"]["content"], {"in_tokens": j.get("prompt_eval_count"),
                                                     "out_tokens": j.get("eval_count")}
                u = j.get("usage") or {}
                return j["choices"][0]["message"]["content"], {"in_tokens": u.get("prompt_tokens"),
                                                               "out_tokens": u.get("completion_tokens")}
            except (requests.RequestException, KeyError, IndexError, ValueError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"{self.backend} call to {self.model} at {self.base_url} failed: {last}")
