"""Thin Groq client for open-source LLMs. Uses the OpenAI-compatible HTTP API.

Reads GROQ_API_KEY from the environment; never persists or logs the key.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("WIKIKGQA_MODEL", "llama-3.3-70b-versatile")


class GroqError(RuntimeError):
    pass


def _get_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise GroqError("GROQ_API_KEY not set in environment")
    return key


def _parse_retry_after(headers, err_body: str) -> float:
    """Honour the server's Retry-After header (seconds), or parse the JSON body for a wait hint."""
    val = headers.get("Retry-After") if headers else None
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    # Groq returns a body like {"error": {"message": "...try again in 12.345s..."}}
    m = re.search(r"try again in ([\d.]+)\s*s", err_body, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 1.0
    return 0.0


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout_s: int = 180,
    retries: int = 8,
) -> str:
    """Single chat completion. Returns the assistant's content string.

    Retries with adaptive backoff on rate-limit (429) and transient 5xx errors.
    Honours the server's Retry-After header when present.
    """
    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode()
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(ENDPOINT, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {_get_key()}")
        req.add_header("User-Agent", "Mozilla/5.0")
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = _parse_retry_after(e.headers, err_body)
                if wait <= 0:
                    wait = min(2 ** attempt + 5, 60)
                wait = min(wait, 200)
                time.sleep(wait)
                last_err = e
                continue
            raise GroqError(f"HTTP {e.code}: {err_body[:300]}") from e
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30))
                last_err = e
                continue
            raise GroqError(f"URL error: {e}") from e
        except (socket.timeout, TimeoutError, ConnectionError) as e:
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30))
                last_err = e
                continue
            raise GroqError(f"Network timeout: {e}") from e
    raise GroqError(f"All retries exhausted: {last_err}")


SPARQL_CODE_BLOCK = re.compile(r"```(?:sparql|SPARQL)?\s*(.*?)```", re.DOTALL)
PREFIX_LINE = re.compile(r"^\s*PREFIX\s+\w+:\s*<[^>]+>\s*$", re.IGNORECASE | re.MULTILINE)


def extract_sparql(text: str) -> str:
    """Best-effort extraction of a SPARQL query from an LLM response.

    Strategy:
      1. If the response contains ```sparql … ``` fenced block(s), use the first.
      2. Else, take everything after the first SELECT/ASK keyword.
      3. Strip leading PREFIX lines (our sparql_client prepends a complete set).
    """
    text = text.strip()
    m = SPARQL_CODE_BLOCK.search(text)
    if m:
        candidate = m.group(1).strip()
    else:
        kw_match = re.search(r"\b(SELECT|ASK)\b", text, re.IGNORECASE)
        candidate = text[kw_match.start():].strip() if kw_match else text
    # Strip any leading PREFIX declarations the model added; we prepend our own
    candidate = PREFIX_LINE.sub("", candidate).strip()
    return candidate
