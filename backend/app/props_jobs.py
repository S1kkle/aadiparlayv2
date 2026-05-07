from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from app.models.core import RankedPropsResponse, SportId


JobStatus = Literal["running", "done", "error"]


@dataclass
class PropsJob:
    id: str
    status: JobStatus = "running"
    created_at_epoch: float = field(default_factory=lambda: time.time())
    updated_at_epoch: float = field(default_factory=lambda: time.time())
    progress: dict[str, Any] = field(default_factory=dict)
    result: RankedPropsResponse | None = None
    error: str | None = None
    events: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)

    async def emit(self, payload: dict[str, Any]) -> None:
        self.updated_at_epoch = time.time()
        await self.events.put(payload)


class PropsJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, PropsJob] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> PropsJob:
        jid = str(uuid.uuid4())
        job = PropsJob(id=jid)
        async with self._lock:
            self._jobs[jid] = job
        await job.emit({"type": "created", "job_id": jid})
        return job

    async def get(self, job_id: str) -> PropsJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def set_done(self, job: PropsJob, result: RankedPropsResponse) -> None:
        job.status = "done"
        job.result = result
        await job.emit({"type": "done", "job_id": job.id})

    async def set_error(self, job: PropsJob, error: str) -> None:
        job.status = "error"
        job.error = error
        await job.emit({"type": "error", "job_id": job.id, "error": error})


def sse_format(payload: dict[str, Any]) -> bytes:
    return (f"data: {json.dumps(payload, ensure_ascii=False)}\n\n").encode("utf-8")


@dataclass(frozen=True)
class PropsJobRequest:
    sport: SportId = "UNKNOWN"
    scope: Literal["all", "featured"] = "all"
    refresh: bool = False
    max_props: int = 10
    ai_limit: int = 10
    require_ai_count: int = 15

