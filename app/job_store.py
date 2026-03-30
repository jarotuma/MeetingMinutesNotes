import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional

_store: dict[str, "Job"] = {}
_lock = asyncio.Lock()


@dataclass
class Job:
    id: str
    status: str = "pending"          # pending | transcribing | summarizing | done | error
    progress_msg: str = "Ve frontě"
    transcript: Optional[str] = None
    summary: Optional[dict] = None
    error: Optional[str] = None


async def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()))
    async with _lock:
        _store[job.id] = job
    return job


async def get_job(job_id: str) -> Optional[Job]:
    return _store.get(job_id)


async def update_job(job_id: str, **kwargs) -> None:
    async with _lock:
        job = _store.get(job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
