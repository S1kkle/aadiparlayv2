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
        async with httpx.AsyncClient(timeout=30) as client:
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

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                self._cfg.over_under_lines_url,
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            return resp.json()

