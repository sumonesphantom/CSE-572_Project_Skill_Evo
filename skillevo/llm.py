"""Chat clients (Ollama, OpenAI-compatible) with a persistent on-disk response cache.

Greedy calls use temperature 0, so a (model, messages) pair always maps to the
same answer. Caching it means the no-skill baseline and unchanged libraries
are never re-solved, which is what makes paired A/B edit tests affordable on a
laptop. Sampled calls (self-consistency) add (temperature, sample index) to the
cache key, so each sample is cached separately and greedy keys are unchanged.
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
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def get_meta(self, key: str) -> dict:
        with self._lock:
            row = self._conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else {}

    def put(self, key: str, value: str, meta: dict | None = None) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
            if meta:
                self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(meta)))
            self._conn.commit()


class OllamaLLM:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 cache: LLMCache | None = None, num_ctx: int = 8192,
                 max_tokens: int = 1024, timeout: int = 600):
        self.model = model          # also the cache namespace
        self.api_model = model      # name sent to the server
        self.host = host.rstrip("/")
        self.cache = cache
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.calls = 0
        self.cache_hits = 0

    def _key(self, messages: list[dict], temperature: float = 0.0, sample: int = 0) -> str:
        d = {"m": self.model, "msgs": messages, "n": self.max_tokens}
        if temperature or sample:  # greedy calls keep their original cache key
            d.update(t=temperature, s=sample)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def chat(self, messages: list[dict], temperature: float = 0.0, sample: int = 0) -> str:
        return self.chat_meta(messages, temperature, sample)[0]

    def chat_meta(self, messages: list[dict], temperature: float = 0.0, sample: int = 0) -> tuple[str, dict]:
        """Like chat(); also returns {cached, seconds, in_tokens, out_tokens} (as first measured, on a cache hit)."""
        key = self._key(messages, temperature, sample)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit, {**self.cache.get_meta(key), "cached": True}
        t0 = time.time()
        text, meta = self._request(messages, temperature, sample)
        meta = {**meta, "seconds": round(time.time() - t0, 3), "cached": False}
        text = strip_think(text)
        self.calls += 1
        if self.cache is not None:
            self.cache.put(key, text, meta)
        return text, meta

    def _request(self, messages, temperature, sample) -> tuple[str, dict]:
        payload = {
            "model": self.api_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "seed": sample, "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens},
        }
        for attempt in range(3):
            try:
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                j = r.json()
                return j["message"]["content"], {"in_tokens": j.get("prompt_eval_count"),
                                                 "out_tokens": j.get("eval_count")}
            except (requests.RequestException, KeyError) as e:
                if attempt == 2:
                    raise RuntimeError(f"Ollama call to {self.api_model} failed: {e}") from e
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")


class OpenAICompatLLM(OllamaLLM):
    """Any OpenAI-style /v1/chat/completions server, e.g. `mlx_lm.server` serving a LoRA-tuned model."""

    def _request(self, messages, temperature, sample) -> tuple[str, dict]:
        payload = {"model": self.api_model, "messages": messages, "temperature": temperature,
                   "max_tokens": self.max_tokens, "seed": sample}
        for attempt in range(3):
            try:
                r = requests.post(f"{self.host}/v1/chat/completions", json=payload, timeout=self.timeout)
                r.raise_for_status()
                j = r.json()
                u = j.get("usage") or {}
                return j["choices"][0]["message"]["content"], {"in_tokens": u.get("prompt_tokens"),
                                                               "out_tokens": u.get("completion_tokens")}
            except (requests.RequestException, KeyError, IndexError) as e:
                if attempt == 2:
                    raise RuntimeError(f"OpenAI-compatible call to {self.api_model} at {self.host} failed: {e}") from e
                time.sleep(2 * (attempt + 1))
        raise AssertionError("unreachable")


def make_llm(model: str, host: str = "http://localhost:11434", cache: LLMCache | None = None, **kw) -> OllamaLLM:
    """`mlx:<label>[@host:port]` -> OpenAICompatLLM (default localhost:8080); anything else -> Ollama.

    <label> only names the cache namespace (use the adapter dir, e.g. mlx:coder7b-r16); requests ask
    the server for "default_model", i.e. whatever `mlx_lm.server --model ... --adapter-path ...` loaded.
    """
    if model.startswith("mlx:"):
        label, _, mhost = model.partition("@")  # host is not part of the cache namespace
        llm = OpenAICompatLLM(label, f"http://{mhost or 'localhost:8080'}", cache, **kw)
        llm.api_model = "default_model"
        return llm
    return OllamaLLM(model, host, cache, **kw)


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
