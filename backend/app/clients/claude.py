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
    "You are a sports prop analyst providing a SMALL CALIBRATED ADJUSTMENT to a "
    "statistical model that already prices the pick. Your job is NOT to reprice the "
    "pick from scratch — it is to nudge the existing probability up or down by a "
    "small amount based on qualitative factors the math model can't see (injuries, "
    "matchup detail, recent context).\n\n"
    "Return ONLY valid JSON with keys: "
    "summary (string), overall_bias (-1|0|1 where 1 = FAVORS pick direction, "
    "-1 = AGAINST, 0 = neutral), confidence (float 0..1), "
    "prob_adjustment (float between -0.05 and +0.05 — bounded; values outside this "
    "range will be clamped). tailwinds (string[]), risk_factors (string[]).\n\n"
    "CALIBRATION RULES (critical — published research shows frontier LLMs are "
    "systematically overconfident on prediction tasks):\n"
    "- Aim for honest calibration: across 100 picks at confidence X, X% should hit. "
    "Most picks SHOULD be near 0.5 — that is healthy.\n"
    "- Reserve 0.80+ for unusual situations with multiple converging strong signals.\n"
    "- prob_adjustment > +0.03 or < -0.03 should require a concrete cited factor "
    "(injury, large rest gap, named matchup edge).\n\n"
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
