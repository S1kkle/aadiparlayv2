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

    _DEFAULT_SYSTEM = (
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

