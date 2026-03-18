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

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

SYSTEM_PROMPT = (
    "You are a sports prop analyst. "
    "Return ONLY valid JSON with keys: "
    "summary (string), overall_bias (-1|0|1 where 1 = FAVORS the pick direction given, "
    "-1 = AGAINST the pick direction given, 0 = neutral), confidence (float 0.0 to 1.0), "
    "prob_adjustment (float between -0.15 and +0.15, your estimated shift to the model probability "
    "based on qualitative factors like injuries, matchup, trend, rest — e.g. +0.05 means 5% more likely), "
    "tailwinds (string[]), risk_factors (string[]).\n\n"
    "CONFIDENCE SCALE (use the FULL range — do NOT default to 0.5):\n"
    "- 0.90-1.0: Very strong conviction — multiple factors strongly align\n"
    "- 0.80-0.89: Strong conviction — solid statistical and contextual support\n"
    "- 0.70-0.79: Moderate conviction — decent support but some uncertainty\n"
    "- 0.50-0.69: Mild lean — could go either way\n"
    "- 0.0-0.49: Low conviction — significant concerns\n"
    "Most good picks should land 0.75-0.90. Differentiate confidently.\n\n"
    "The summary must be 2-4 sentences, referencing matchup context and "
    "citing at least two numbers from the input (line, avg, hit rate, model_prob, edge). "
    "If you mention injuries, ONLY reference names that appear in the provided data. "
    "Do NOT invent injuries."
)


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
        self, *, prompt: str, timeout_s: float = 45.0, system: str | None = None
    ) -> dict[str, Any]:
        payload = {
            "model": self._cfg.model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "system": system or SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }

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
