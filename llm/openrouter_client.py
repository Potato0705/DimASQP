"""Minimal OpenRouter chat/completions client.

Design goals:
  - Only depends on `requests` (already in requirements.txt) — no new deps.
  - API key via env var OPENROUTER_API_KEY (never written to disk).
  - Exponential-backoff retries on transient failures (429, 5xx, timeouts).
  - Returns the raw assistant message string; the caller decides how to parse.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from .config import DEFAULTS


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter call fails after all retries."""


class OpenRouterClient:
    """Thin wrapper around POST {base_url}/chat/completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_base_s: Optional[float] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. Export it in your shell or use a .env loader."
            )
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULTS["base_url"]).rstrip("/")
        self.timeout_s = float(timeout_s if timeout_s is not None else DEFAULTS["timeout_s"])
        self.max_retries = int(max_retries if max_retries is not None else DEFAULTS["max_retries"])
        self.backoff_base_s = float(backoff_base_s if backoff_base_s is not None else DEFAULTS["backoff_base_s"])
        self.http_referer = http_referer or DEFAULTS.get("http_referer", "")
        self.x_title = x_title or DEFAULTS.get("x_title", "")

    # ------------------------------------------------------------------ public
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = DEFAULTS["temperature"],
        top_p: float = DEFAULTS["top_p"],
        max_tokens: int = DEFAULTS["max_tokens"],
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a chat-completion request.

        Returns a dict with keys:
          - 'content':       assistant text (may be empty string)
          - 'usage':         dict of prompt/completion/total tokens (may be None)
          - 'model':         the model tag actually served (may differ from request)
          - 'raw':           the full parsed JSON response
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        url = f"{self.base_url}/chat/completions"

        last_err: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = f"network: {e!r}"
                self._sleep_backoff(attempt)
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as e:
                    raise OpenRouterError(f"non-JSON 200 response: {e!r}: {resp.text[:200]}")
                content = ""
                try:
                    content = data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    content = ""
                return {
                    "content": content,
                    "usage": data.get("usage"),
                    "model": data.get("model", model),
                    "raw": data,
                }

            # Retryable statuses: 408 / 409 / 429 / 500 / 502 / 503 / 504
            if resp.status_code in (408, 409, 429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                self._sleep_backoff(attempt)
                continue

            # Non-retryable error
            raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        raise OpenRouterError(f"exhausted {self.max_retries + 1} attempts; last error: {last_err}")

    # ---------------------------------------------------------------- internal
    def _sleep_backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff (capped implicitly by max_retries)."""
        if attempt >= self.max_retries:
            return
        delay = self.backoff_base_s * (2 ** attempt)
        time.sleep(delay)


def parse_json_content(content: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parser for LLM output.

    Handles two common failure modes:
      1. Content wrapped in ```json ... ``` code fences.
      2. Trailing prose after the JSON object.

    Returns None if no JSON object can be extracted.
    """
    if not content:
        return None
    text = content.strip()

    # Strip code fences.
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Fast path: direct parse.
    try:
        return json.loads(text)
    except ValueError:
        pass

    # Slow path: find the outermost {...} block.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except ValueError:
                    return None
    return None
