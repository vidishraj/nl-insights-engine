"""The job manager — async submit / stream / cancel over the sync pipelines.

Concurrency story (the brief asks for this explicitly):
- Each job's blocking work runs in a worker thread (``asyncio.to_thread``), so many
  datasets ingest at once and the event loop never blocks. One DuckDB file per dataset
  means cross-dataset work never contends.
- Two ingests of the SAME dataset serialise on a per-dataset lock, and the second's
  atomic promote simply wins — no corruption, no half state.
- A query against a dataset mid-re-ingest reads the previous promoted file (or a clean
  404 if it never had one); it never sees the half-built staging file.

Partial failure & cancellation:
- Ingest builds to staging and promotes atomically; a job that dies leaves no queryable
  half-dataset (the staging file is removed in a finally).
- Cancellation is cooperative: ``cancel`` sets a flag the ``emit`` callback checks at
  each stage boundary and raises on, so it actually stops rather than being ignored.

Idempotency:
- A repeated ingest of identical content (same idempotency key) returns the SAME job
  instead of ingesting twice — the retried-request lesson, applied directly.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..ingestion.loader import dataset_id_for
from ..interpreter.ir import QueryIR
from ..provider import LLMProvider
from .events import TERMINAL_STATES, JobError, JobEvent, JobKind, JobState
from .pipeline import Cancelled, PipelineError, QueryResult, run_ingest, run_query
from .store import Dataset, DatasetStore


class JobView(BaseModel):
    """The serialisable snapshot of a job the API returns (poll fallback + terminal)."""

    id: str
    kind: JobKind
    state: JobState
    dataset_id: str | None = None
    events: list[JobEvent] = []
    error: JobError | None = None
    result: dict[str, Any] | None = None


class _Job:
    """Internal job record — holds asyncio/threading primitives, so not a pydantic model."""

    def __init__(self, kind: JobKind) -> None:
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.state = JobState.QUEUED
        self.events: list[JobEvent] = []
        self.error: JobError | None = None
        self.result_obj: Dataset | QueryResult | None = None
        self.dataset_id: str | None = None
        self._seq = 0
        self._subscribers: set[asyncio.Queue[JobEvent | None]] = set()
        self._done = asyncio.Event()
        self._cancel = threading.Event()
        self._task: asyncio.Task[None] | None = None

    def view(self) -> JobView:
        return JobView(
            id=self.id,
            kind=self.kind,
            state=self.state,
            dataset_id=self.dataset_id,
            events=list(self.events),
            error=self.error,
            result=self._result_summary(),
        )

    def _result_summary(self) -> dict[str, Any] | None:
        if isinstance(self.result_obj, Dataset):
            d = self.result_obj
            return {
                "dataset_id": d.dataset_id,
                "row_count": d.row_count,
                "measures": [m.name for m in d.model.measures if m.available],
            }
        if isinstance(self.result_obj, QueryResult):
            return self.result_obj.model_dump(mode="json")
        return None


class JobManager:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        data_dir: Path,
        store: DatasetStore,
    ) -> None:
        self._provider = provider
        self._model = model
        self._data_dir = data_dir
        self._staging_dir = data_dir / ".staging"
        self._store = store
        self._jobs: dict[str, _Job] = {}
        self._idempotency: dict[str, str] = {}  # key -> job id
        self._locks: dict[str, asyncio.Lock] = {}

    # -- accessors ---------------------------------------------------------------
    def get(self, job_id: str) -> JobView | None:
        job = self._jobs.get(job_id)
        return job.view() if job else None

    def query_result(self, job_id: str) -> QueryResult | None:
        job = self._jobs.get(job_id)
        return job.result_obj if job and isinstance(job.result_obj, QueryResult) else None

    def cancel(self, job_id: str) -> bool:
        """Request cancellation. Takes effect at the next stage boundary. Returns False
        for an unknown or already-terminal job."""
        job = self._jobs.get(job_id)
        if job is None or job.state in TERMINAL_STATES:
            return False
        job._cancel.set()
        return True

    # -- submission --------------------------------------------------------------
    def submit_ingest(
        self,
        *,
        source: Path,
        name: str | None,
        idempotency_key: str,
        display_name: str | None = None,
    ) -> JobView:
        # Idempotent: an identical in-flight or succeeded ingest is reused, never redone.
        existing_id = self._idempotency.get(idempotency_key)
        if existing_id is not None:
            existing = self._jobs[existing_id]
            if existing.state not in {JobState.FAILED, JobState.CANCELLED}:
                return existing.view()

        job = _Job(JobKind.INGEST)
        job.dataset_id = dataset_id_for(source, name)
        self._jobs[job.id] = job
        self._idempotency[idempotency_key] = job.id

        async def work() -> None:
            # Serialise on the DATASET id (not the content key): two ingests of the same
            # dataset share one staging file, so they must not run concurrently.
            lock = self._lock_for(job.dataset_id or job.id)
            async with lock:
                await self._run(
                    job,
                    lambda emit: run_ingest(
                        source=source,
                        name=name,
                        provider=self._provider,
                        model=self._model,
                        data_dir=self._data_dir,
                        staging_dir=self._staging_dir,
                        emit=emit,
                        display_name=display_name,
                    ),
                    on_success=self._register_dataset,
                )

        job._task = asyncio.get_running_loop().create_task(work())
        return job.view()

    def submit_query(
        self, *, dataset: Dataset, question: str, previous_ir: QueryIR | None
    ) -> JobView:
        job = _Job(JobKind.QUERY)
        job.dataset_id = dataset.dataset_id
        self._jobs[job.id] = job

        async def work() -> None:
            await self._run(
                job,
                lambda emit: run_query(
                    dataset=dataset,
                    question=question,
                    previous_ir=previous_ir,
                    provider=self._provider,
                    model=self._model,
                    emit=emit,
                ),
            )

        job._task = asyncio.get_running_loop().create_task(work())
        return job.view()

    # -- streaming ---------------------------------------------------------------
    async def stream(self, job_id: str) -> AsyncIterator[JobEvent]:
        """Yield this job's events — history first, then live — until it terminates."""
        job = self._jobs[job_id]
        q: asyncio.Queue[JobEvent | None] = asyncio.Queue()
        job._subscribers.add(q)  # subscribe BEFORE snapshotting (no await between)
        try:
            history = list(job.events)
            done_already = job._done.is_set()
            for ev in history:
                yield ev
            last = history[-1].seq if history else -1
            if done_already:
                return
            while True:
                item = await q.get()
                if item is None:  # terminal sentinel
                    return
                if item.seq > last:
                    yield item
        finally:
            job._subscribers.discard(q)

    async def wait(self, job_id: str) -> JobView:
        """Await terminal state (used by tests and the non-streaming callers)."""
        job = self._jobs[job_id]
        await job._done.wait()
        return job.view()

    # -- internals ---------------------------------------------------------------
    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _register_dataset(self, job: _Job, result: Dataset) -> None:
        job.dataset_id = result.dataset_id
        self._store.put(result)

    async def _run(
        self,
        job: _Job,
        work: Callable[[Any], Any],
        *,
        on_success: Callable[[_Job, Any], None] | None = None,
    ) -> None:
        job.state = JobState.RUNNING
        emit = self._make_emit(job)
        try:
            result = await asyncio.to_thread(work, emit)
            job.result_obj = result
            if on_success is not None:
                on_success(job, result)
            job.state = JobState.SUCCEEDED
        except Cancelled:
            job.state = JobState.CANCELLED
            job.error = JobError(code="cancelled", message="job cancelled by request")
        except PipelineError as exc:
            job.state = JobState.FAILED
            job.error = exc.error
        except Exception:  # never leak a stack trace to the caller
            job.state = JobState.FAILED
            job.error = JobError(code="internal_error", message="an unexpected error occurred")
        finally:
            self._finish(job)

    def _make_emit(self, job: _Job) -> Callable[[str, str], None]:
        loop = asyncio.get_running_loop()

        def emit(stage: str, message: str) -> None:
            # Called from the worker thread. Check cancellation FIRST so a cancel lands
            # at this boundary, then publish the progress event onto the loop thread.
            if job._cancel.is_set():
                raise Cancelled()
            loop.call_soon_threadsafe(self._publish, job, stage, message)

        return emit

    def _publish(self, job: _Job, stage: str, message: str) -> None:
        ev = JobEvent(seq=job._seq, stage=stage, message=message)
        job._seq += 1
        job.events.append(ev)
        for q in list(job._subscribers):
            q.put_nowait(ev)

    def _finish(self, job: _Job) -> None:
        job._done.set()
        for q in list(job._subscribers):
            q.put_nowait(None)  # unblock streams so they can send the terminal frame


__all__ = [
    "DatasetStore",
    "JobManager",
    "JobView",
    "QueryResult",
]
