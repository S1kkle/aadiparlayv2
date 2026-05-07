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

from app.clients._prompts import PROP_SYSTEM_PROMPT

GROQ_BASE = "https://api.groq.com/openai/v1"
SYSTEM_JSON = PROP_SYSTEM_PROMPT


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

    async def analyze_prop(
        self,
        *,
        prompt: str,
        timeout_s: float = 30.0,
        system: str | None = None,
        json_mode: bool = True,
    ) -> dict[str, Any]:
        url = f"{GROQ_BASE}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system or SYSTEM_JSON},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        # Groq supports OpenAI-compatible response_format. Forcing json_object
        # eliminates the "did not return valid JSON" failure mode in
        # production. Disable when the caller's prompt expects free-form text.
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
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
