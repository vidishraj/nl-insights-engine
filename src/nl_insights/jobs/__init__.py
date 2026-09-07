"""Job orchestrator — async submit / stream (SSE) / cancel for ingestion and queries.

Owns job ids, progress events, the dataset registry, atomic promotion (crash safety),
per-dataset serialisation, idempotent submission, and cooperative cancellation. The
HTTP layer is a thin adapter over this; all the concurrency/failure behaviour lives
here so it can be tested without a server.
"""

from .events import JobError, JobEvent, JobKind, JobState
from .manager import JobManager, JobView
from .pipeline import Cancelled, PipelineError, QueryResult
from .store import Dataset, DatasetStore, load_persisted, write_sidecar

__all__ = [
    "Cancelled",
    "Dataset",
    "DatasetStore",
    "load_persisted",
    "write_sidecar",
    "JobError",
    "JobEvent",
    "JobKind",
    "JobManager",
    "JobState",
    "JobView",
    "PipelineError",
    "QueryResult",
]
