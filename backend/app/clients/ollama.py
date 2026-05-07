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

    @property
    def _DEFAULT_SYSTEM(self) -> str:
        # Defer import to avoid hard dependency at module-load time.
        from app.clients._prompts import PROP_SYSTEM_PROMPT
        return PROP_SYSTEM_PROMPT

    async def analyze_prop(self, *, prompt: str, timeout_s: float = 30.0, system: str | None = None) -> dict[str, Any]:
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
                    "content": system or self._DEFAULT_SYSTEM,
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

