"""Ollama chat client with a persistent on-disk response cache.

All calls use temperature 0, so a (model, messages) pair always maps to the
same answer. Caching it means the no-skill baseline and unchanged libraries
are never re-solved, which is what makes paired A/B edit tests affordable on a
laptop.
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


class LLMCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
            self._conn.commit()


class OllamaLLM:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 cache: LLMCache | None = None, num_ctx: int = 8192,
                 max_tokens: int = 1024, timeout: int = 600, think: bool | None = False):
        self.model = model
        self.host = host.rstrip("/")
        self.cache = cache
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.think = think          # None: do not send Ollama's think flag at all
        self.calls = 0
        self.cache_hits = 0

    def _key(self, messages: list[dict]) -> str:
        blob = json.dumps({"m": self.model, "msgs": messages, "n": self.max_tokens}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict]) -> str:
        key = self._key(messages)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens},
        }
        if self.think is not None:
            payload["think"] = self.think
        for attempt in range(3):
            try:
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                text = r.json()["message"]["content"]
                break
            except (requests.RequestException, KeyError) as e:
                if attempt == 2:
                    raise RuntimeError(f"Ollama call to {self.model} failed: {e}") from e
                time.sleep(2 * (attempt + 1))
        text = strip_think(text)
        self.calls += 1
        if self.cache is not None:
            self.cache.put(key, text)
        return text


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_sql(text: str) -> str:
    blocks = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.S | re.I)
    sql = blocks[-1] if blocks else text
    return sql.strip().rstrip(";").strip()


def extract_json(text: str):
    """Pull the first JSON object/array out of a model reply."""
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    candidates = fenced + [text]
    for c in candidates:
        c = c.strip()
        for opener, closer in (("[", "]"), ("{", "}")):
            i, j = c.find(opener), c.rfind(closer)
            if i != -1 and j > i:
                try:
                    return json.loads(c[i:j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON found in model output")
