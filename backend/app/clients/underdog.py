from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import httpx


@dataclass(frozen=True)
class UnderdogConfig:
    over_under_lines_url: str
    auth_token: str | None
    user_location_token: str | None
    product: str | None
    product_experience_id: str | None
    state_config_id: str | None


class UnderdogClient:
    def __init__(self, cfg: UnderdogConfig) -> None:
        self._cfg = cfg
        # Long-lived AsyncClient with connection pooling. Previously we
        # opened a fresh client per call which forced a new TCP/TLS
        # handshake on every fetch — measurable overhead given the
        # over-under endpoint is hit on every prop load and every CLV cycle.
        self._http: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=30,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=8),
                http2=False,
            )
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "aadiparlayv2/1.0",
        }
        if self._cfg.auth_token:
            headers["Authorization"] = self._cfg.auth_token
        if self._cfg.user_location_token:
            headers["User-Location-Token"] = self._cfg.user_location_token
        return headers

    def _base_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self._cfg.product:
            params["product"] = self._cfg.product
        if self._cfg.product_experience_id:
            params["product_experience_id"] = self._cfg.product_experience_id
        if self._cfg.state_config_id:
            params["state_config_id"] = self._cfg.state_config_id
        return params

    async def fetch_all_over_under_lines(self) -> dict[str, Any]:
        """
        Best-effort: some Underdog deployments return all pregame lines when called without IDs.
        """
        client = await self._get_client()
        resp = await client.get(
            self._cfg.over_under_lines_url,
            headers=self._headers(),
            params=self._base_params(),
        )
        resp.raise_for_status()
        return resp.json()

    async def fetch_over_under_lines_by_ids(self, over_under_ids: Iterable[str]) -> dict[str, Any]:
        ids = list(over_under_ids)
        if not ids:
            return await self.fetch_all_over_under_lines()

        params = self._base_params()
        params["over_under_ids"] = ",".join(ids)

        client = await self._get_client()
        resp = await client.get(
            self._cfg.over_under_lines_url,
            headers=self._headers(),
            params=params,
        )
        resp.raise_for_status()
        return resp.json()

