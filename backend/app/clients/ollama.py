from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import json


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str  # e.g. http://localhost:11434
    model: str


class OllamaClient:
    def __init__(self, cfg: OllamaConfig) -> None:
        # On some Windows setups, "localhost" resolves to IPv6 ::1 first while Ollama
        # only listens on 127.0.0.1, causing availability checks to fail.
        base = (cfg.base_url or "").strip()
        if "://localhost" in base:
            base = base.replace("://localhost", "://127.0.0.1")
        self._cfg = OllamaConfig(base_url=base, model=cfg.model)

    async def is_available(self) -> bool:
        url = f"{self._cfg.base_url.rstrip('/')}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(url)
                return resp.status_code == 200
        except Exception:
            return False

    async def analyze_prop(self, *, prompt: str, timeout_s: float = 30.0) -> dict[str, Any]:
        """
        Calls Ollama chat endpoint expecting strict JSON output.
        """
        url = f"{self._cfg.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self._cfg.model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a sports prop analyst. "
                        "Return ONLY valid JSON with keys: "
                        "summary (string), overall_bias (-1|0|1), confidence (0..1), "
                        "tailwinds (string[]), risk_factors (string[]). "
                        "The summary must be 2-4 sentences, explicitly referencing matchup context and injuries/availability "
                        "when provided, and must cite at least two numbers from the input (e.g., line, last10 avg/hit rate, model_prob, edge). "
                        "If you mention injuries, ONLY reference names/lines that appear in the provided TEAM_INJURY_LINES / OPP_INJURY_LINES, "
                        "and copy the injury line(s) verbatim in parentheses. Do NOT invent injuries."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.2},
        }
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = (data.get("message") or {}).get("content")
            if not isinstance(msg, str):
                raise ValueError("Unexpected Ollama response shape (no message.content).")
            # Ollama returns JSON as a string in content when format=json
            try:
                return json.loads(msg)
            except Exception as e:
                raise ValueError(f"Ollama did not return valid JSON content: {e}") from e

