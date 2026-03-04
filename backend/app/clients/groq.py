"""
Groq cloud LLM client (OpenAI-compatible API).
Use when GROQ_API_KEY is set for always-on hosting without local Ollama.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

GROQ_BASE = "https://api.groq.com/openai/v1"
SYSTEM_JSON = (
    "You are a sports prop analyst. "
    "Return ONLY valid JSON with keys: "
    "summary (string), overall_bias (-1|0|1), confidence (0..1), "
    "tailwinds (string[]), risk_factors (string[]). "
    "The summary must be 2-4 sentences, explicitly referencing matchup context and injuries/availability "
    "when provided, and must cite at least two numbers from the input (e.g., line, last10 avg/hit rate, model_prob, edge). "
    "If you mention injuries, ONLY reference names/lines that appear in the provided TEAM_INJURY_LINES / OPP_INJURY_LINES, "
    "and copy the injury line(s) verbatim in parentheses. Do NOT invent injuries."
)


@dataclass(frozen=True)
class GroqConfig:
    api_key: str
    model: str  # e.g. llama-3.1-8b-instant, mixtral-8x7b-32768


class GroqClient:
    """Cloud LLM via Groq (same interface as OllamaClient for ranker)."""

    def __init__(self, cfg: GroqConfig) -> None:
        self._cfg = cfg

    async def is_available(self) -> bool:
        if not (self._cfg.api_key or "").strip():
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{GROQ_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._cfg.api_key.strip()}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._cfg.model,
                        "messages": [{"role": "user", "content": "Say OK"}],
                        "max_tokens": 4,
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def analyze_prop(self, *, prompt: str, timeout_s: float = 30.0) -> dict[str, Any]:
        url = f"{GROQ_BASE}/chat/completions"
        payload = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_JSON},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key.strip()}",
            "Content-Type": "application/json",
        }
        max_retries = 3
        data: dict[str, Any] = {}
        for attempt in range(max_retries):
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 429:
                    wait = min(2 ** (attempt + 1), 15)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
        else:
            raise RuntimeError("Groq rate limit (429) after retries")
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("Groq response had no choices")
        msg = (choices[0].get("message") or {}).get("content")
        if not isinstance(msg, str):
            raise ValueError("Groq response message.content missing or not string")
        # Groq may wrap JSON in markdown code block; strip if present
        raw = msg.strip()
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
            raise ValueError(f"Groq did not return valid JSON: {e}") from e
