"""Prompt-response cache for reproducible LLM experiments."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


Provider = Callable[[str, str, dict[str, Any]], str]
DEFAULT_KEY_FILES = {
    "OPENAI_API_KEY": Path("paper-proposal-2/keys/openai.md"),
    "GEMINI_API_KEY": Path("paper-proposal-2/keys/gemini.md"),
}


class LLMCache:
    def __init__(
        self,
        cache_dir: Path,
        provider_name: str = "stub",
        model: str = "stub",
        provider: Provider | None = None,
        cache_mode: str = "live",
        allow_stub: bool = True,
    ):
        if provider_name not in {"stub", "openai", "gemini"}:
            raise ValueError(f"unknown LLM provider {provider_name!r}")
        if cache_mode not in {"live", "replay"}:
            raise ValueError(f"unknown cache mode {cache_mode!r}")
        self.cache_dir = cache_dir
        self.provider_name = provider_name
        self.model = model
        self.provider = provider or self._provider_for(provider_name)
        self.cache_mode = cache_mode
        self.allow_stub = allow_stub
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, prompt: str, model: str, params: dict[str, Any] | None = None) -> str:
        payload = json.dumps(
            {"provider": self.provider_name, "prompt": prompt, "model": model, "params": params or {}},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(self, prompt: str, model: str | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        model = model or self.model
        key = self.key(prompt, model, params)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.cache_mode == "replay":
            raise RuntimeError(f"LLM cache miss for provider={self.provider_name} model={model} key={key}")
        if self.provider is not None:
            text = self.provider(prompt, model, params)
        elif self.allow_stub:
            text = self._stub(prompt)
        else:
            raise RuntimeError(f"LLM cache miss for provider={self.provider_name} model={model} key={key}")
        record = {
            "key": key,
            "provider": self.provider_name,
            "model": model,
            "params": params,
            "prompt": prompt,
            "text": text,
            "prompt_tokens": max(1, len(prompt.split())),
            "completion_tokens": max(1, len(text.split())),
        }
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return record

    @staticmethod
    def _provider_for(provider_name: str) -> Provider | None:
        if provider_name == "stub":
            return None
        if provider_name == "openai":
            return _openai_complete
        if provider_name == "gemini":
            return _gemini_complete
        raise ValueError(f"unknown LLM provider {provider_name!r}")

    @staticmethod
    def _stub(prompt: str) -> str:
        return json.dumps(
            {
                "instructions": [
                    {
                        "target": "entrypoint",
                        "priority": "high",
                        "partition": "boundary",
                        "values": [0, 1, -1],
                        "expected": "execute without unexpected exception",
                    }
                ],
                "source": "stub",
            }
        )


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], retries: int = 3) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"LLM provider request failed with HTTP {exc.code}: {detail}") from None
        except URLError as exc:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"LLM provider request failed: {exc.reason}") from None
    raise RuntimeError("LLM provider request failed after retries")


def _read_key_from_file(env_name: str) -> str | None:
    path = DEFAULT_KEY_FILES.get(env_name)
    if path is None or not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == env_name:
                return value.strip().strip('"').strip("'") or None
            continue
        return line.strip('"').strip("'")
    return None


def _api_key(env_name: str) -> str:
    key = os.environ.get(env_name) or _read_key_from_file(env_name)
    if not key:
        path = DEFAULT_KEY_FILES.get(env_name)
        suffix = f" or {path}" if path is not None else ""
        raise RuntimeError(f"{env_name} is not set; provide it via environment{suffix}")
    return key


def _openai_complete(prompt: str, model: str, params: dict[str, Any]) -> str:
    api_key = _api_key("OPENAI_API_KEY")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params.get("temperature", 0),
        "max_tokens": params.get("max_tokens", 256),
    }
    data = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        payload,
    )
    return str(data["choices"][0]["message"]["content"])


def _gemini_complete(prompt: str, model: str, params: dict[str, Any]) -> str:
    api_key = _api_key("GEMINI_API_KEY")
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": params.get("temperature", 0), "maxOutputTokens": params.get("max_tokens", 256)},
    }
    data = _post_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        {"Content-Type": "application/json"},
        payload,
    )
    parts = data["candidates"][0]["content"].get("parts", [])
    return "".join(str(part.get("text", "")) for part in parts)
