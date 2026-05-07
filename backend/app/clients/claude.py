"""
Anthropic Claude LLM client.
Preferred when ANTHROPIC_API_KEY is set — smarter reasoning and higher rate limits than Groq free tier.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

from app.clients._prompts import (
    ANTHROPIC_PROP_TOOL,
    PROP_SYSTEM_PROMPT,
)

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

SYSTEM_PROMPT = PROP_SYSTEM_PROMPT


@dataclass(frozen=True)
class ClaudeConfig:
    api_key: str
    model: str


class ClaudeClient:
    """Anthropic Claude LLM — same interface as OllamaClient / GroqClient."""

    def __init__(self, cfg: ClaudeConfig) -> None:
        self._cfg = cfg
        self._http: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=60,
                limits=httpx.Limits(max_connections=15, max_keepalive_connections=10),
            )
        return self._http

    async def is_available(self) -> bool:
        if not (self._cfg.api_key or "").strip():
            return False
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{ANTHROPIC_BASE}/messages",
                headers=self._headers(),
                json={
                    "model": self._cfg.model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Say OK"}],
                },
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._cfg.api_key.strip(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    async def analyze_prop(
        self,
        *,
        prompt: str,
        timeout_s: float = 45.0,
        system: str | None = None,
        use_tool: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "system": system or SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Anthropic Tool Use — forces structured output that matches the
        # prop_analysis schema, eliminating "did not return valid JSON" errors
        # that plague free-form prompts. Disable when the caller's prompt
        # uses a different schema (parlay summary, AI selection, etc.).
        if use_tool:
            payload["tools"] = [ANTHROPIC_PROP_TOOL]
            payload["tool_choice"] = {"type": "tool", "name": ANTHROPIC_PROP_TOOL["name"]}

        client = await self._get_client()
        max_retries = 3
        data: dict[str, Any] = {}

        for attempt in range(max_retries):
            resp = await client.post(
                f"{ANTHROPIC_BASE}/messages",
                headers=self._headers(),
                json=payload,
                timeout=timeout_s,
            )
            if resp.status_code == 429:
                wait = min(2 ** (attempt + 1), 15)
                await asyncio.sleep(wait)
                continue
            if resp.status_code == 529:
                await asyncio.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        else:
            raise RuntimeError("Claude rate limit after retries")

        content_blocks = data.get("content") or []
        if not content_blocks:
            raise ValueError("Claude response had no content blocks")

        # Prefer tool_use block (structured) over text block.
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                inp = block.get("input")
                if isinstance(inp, dict):
                    return inp

        raw_text = ""
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                raw_text += block.get("text", "")

        if not raw_text.strip():
            raise ValueError("Claude returned empty text")

        raw = raw_text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Claude did not return valid JSON: {e}") from e
