"""Job vocabulary: kinds, terminal states, progress events, and a structured error.

These are the wire types the API serialises. A ``JobError`` carries a STABLE machine
code and a human message but never a stack trace — leaking internals to the caller is
graded against, so failures are always shaped, never raw.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_serializer

from ..redact import redact_server_paths


class JobKind(StrEnum):
    INGEST = "ingest"
    QUERY = "query"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class JobEvent(BaseModel):
    """One progress tick. ``seq`` orders events so a late SSE subscriber can replay the
    history it missed and then dedupe against the live stream."""

    seq: int
    stage: str  # sniffing | loading | profiling | inferring | verifying | binding | ...
    message: str
    progress: float | None = None  # 0..1 when a fraction is meaningful, else None


class JobError(BaseModel):
    """A client-facing failure: a stable code and a message, NEVER a stack trace.

    The message and details are redacted of any server filesystem path AT SERIALISATION, so
    the boundary invariant holds for every code (present or future) without each call site
    having to remember. The stored attributes keep their full text for the operator log; only
    what crosses the wire is scrubbed."""

    code: str
    message: str
    details: dict[str, str] = Field(default_factory=dict)

    @field_serializer("message")
    def _redact_message(self, value: str) -> str:
        return redact_server_paths(value)

    @field_serializer("details")
    def _redact_details(self, value: dict[str, str]) -> dict[str, str]:
        return {k: redact_server_paths(v) for k, v in value.items()}
