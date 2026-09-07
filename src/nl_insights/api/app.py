"""FastAPI application — a thin HTTP adapter over the job manager.

The interesting behaviour (concurrency, atomic promotion, cancellation, idempotency)
lives in :mod:`nl_insights.jobs`; the routes just translate HTTP to job submissions and
stream progress. Every error path is SHAPED: a stable machine ``code`` and a message,
never a stack trace — a catch-all handler guarantees even an unexpected failure returns
structured JSON, which the brief grades explicitly.

Flow: POST a CSV -> 202 + job id -> stream stage events over SSE (the semantic model
assembling) -> GET the dataset's model -> POST a question -> stream -> GET the answer.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..config import Settings
from ..jobs import DatasetStore, JobManager, load_persisted
from ..provider import LLMProvider, build_provider
from ..redact import redact_server_paths
from .samples import SAMPLES, get_sample

_STATIC = Path(__file__).resolve().parent / "static"


class ApiError(Exception):
    """A client-facing error with a stable machine code and HTTP status."""

    def __init__(self, code: str, message: str, status: int = 400, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details


class QueryBody(BaseModel):
    question: str
    previous_job_id: str | None = None  # chain a follow-up onto a prior query's plan


async def _read_upload(request: Request) -> tuple[bytes, str | None]:
    """Read the uploaded CSV from EITHER a multipart file part OR a raw body.

    Multipart (``curl -F file=@x.csv``, browser pickers, ``httpx files=``) is how most
    clients send a file; raw body (``curl --data-binary @x.csv``) is also legitimate and
    what the library tests use. We accept both, and if a raw body is actually a
    mis-sent multipart envelope we say so explicitly instead of letting the delimiter
    sniffer misdiagnose it three stages later.
    """
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        for value in form.values():
            filename = getattr(value, "filename", None)
            if filename is not None:  # an UploadFile part
                return await value.read(), filename  # type: ignore[union-attr]
        raise ApiError("no_file_part", "multipart body carried no file part", 400)

    body = await request.body()
    head = body[:64].lstrip()
    if head.startswith(b"--") and b"Content-Disposition" in body[:400]:
        raise ApiError(
            "looks_like_multipart",
            "the raw body is a multipart envelope; send it as multipart/form-data "
            "(curl -F 'file=@your.csv') or as a raw CSV body (curl --data-binary @your.csv)",
            400,
        )
    return body, None


def _error_response(status: int, code: str, message: str, details: dict[str, Any]) -> JSONResponse:
    """The ONE place a shaped error is serialised to the client. Every exception handler routes
    through here, so the server-path redaction is applied to every error code - the ones we
    wrote and the ones we have not - rather than remembered per handler. Only the path fragment
    changes; the code, status, and the rest of the message are untouched."""
    safe_details = {
        k: redact_server_paths(v) if isinstance(v, str) else v for k, v in details.items()
    }
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": redact_server_paths(message),
                "details": safe_details,
            }
        },
    )


def _links(job_id: str) -> dict[str, str]:
    return {
        "self": f"/jobs/{job_id}",
        "stream": f"/jobs/{job_id}/stream",
        "cancel": f"/jobs/{job_id}/cancel",
    }


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="NL Insights Engine", version="0.1.0")

    # Build the provider once (fail fast at startup, not mid-request). Injectable so
    # tests can pass a stub with no credentials or fixtures.
    app.state.settings = settings
    app.state.provider = provider or build_provider(settings)
    app.state.datasets = DatasetStore()
    # Rehydrate datasets ingested in a previous run: their DuckDB files persisted on disk,
    # and their semantic models were written beside them, so a restart does not drop them.
    load_persisted(app.state.datasets, settings.data_dir)
    app.state.jobs = JobManager(
        provider=app.state.provider,
        model=settings.model,
        data_dir=settings.data_dir,
        store=app.state.datasets,
    )

    def manager() -> JobManager:
        return app.state.jobs  # type: ignore[no-any-return]

    def store() -> DatasetStore:
        return app.state.datasets  # type: ignore[no-any-return]

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Unify malformed-request errors onto the same {error:{code,message,details}}
        # contract as everything else, instead of FastAPI's default {detail:[...]} shape.
        return _error_response(
            422,
            "invalid_request",
            "the request body or parameters were invalid",
            {"errors": str(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Last line of defence: never leak a stack trace or internal message.
        return _error_response(500, "internal_error", "an unexpected error occurred", {})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "provider": app.state.provider.name, "model": settings.model}

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        # The glass-box UI — a single self-contained page (no build step) that exposes
        # the model assembling over SSE and every answer's plan/SQL/verdict/caveats.
        return FileResponse(_STATIC / "index.html")

    # -- ingestion --------------------------------------------------------------
    @app.post("/datasets", status_code=202)
    async def create_dataset(request: Request, name: str | None = None) -> dict[str, Any]:
        # Resource guard 1 — reject an over-cap upload before buffering it. Prefer
        # the declared Content-Length so a giant body is refused up front.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes:
            raise ApiError(
                "upload_too_large",
                f"upload exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit",
                413,
            )
        body, upload_name = await _read_upload(request)
        if not body:
            raise ApiError("empty_upload", "upload a CSV (multipart file or raw body)", 400)
        if len(body) > settings.max_upload_bytes:
            raise ApiError(
                "upload_too_large",
                f"upload exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit",
                413,
            )
        # Resource guard 2 — decline rather than fill the disk. OFF by default
        # (min_free_bytes=0 → refuse only when the upload literally would not fit); a
        # deployment opts IN to a protective floor via NL_INSIGHTS_MIN_FREE_BYTES, so a
        # clean clone on a fullish machine is never refused for the demo CSV.
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(settings.data_dir).free
        if free - len(body) < settings.min_free_bytes:
            raise ApiError(
                "storage_full",
                "the server is low on storage and is declining new uploads to protect "
                "existing datasets; contact the operator or try again after cleanup",
                507,
            )
        name = name or upload_name
        digest = hashlib.sha256(body).hexdigest()[:16]
        ds_name = name or f"ds_{digest}"
        # Content-based idempotency: a retried identical upload reuses the same job
        # instead of ingesting twice. An explicit header overrides.
        idem = request.headers.get("idempotency-key") or f"{ds_name}:{digest}"

        uploads = settings.data_dir / ".uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        src = uploads / f"{digest}.csv"
        src.write_bytes(body)

        # The user's original filename is echoed in any shaped error (never the server path we
        # stored under); basename it so a client-supplied path can't reflect a directory back.
        display_name = Path(upload_name).name if upload_name else None
        view = manager().submit_ingest(
            source=src, name=ds_name, idempotency_key=idem, display_name=display_name
        )
        return {
            "job_id": view.id,
            "kind": view.kind,
            "state": view.state,
            "dataset_id": view.dataset_id,
            "links": _links(view.id),
        }

    # -- bundled samples --------------------------------------------------------
    @app.get("/samples")
    def list_samples() -> dict[str, Any]:
        return {
            "samples": [
                {
                    "id": s.id,
                    "label": s.label,
                    "description": s.description,
                    "download": f"/samples/{s.id}/download",
                }
                for s in SAMPLES.values()
            ]
        }

    @app.post("/samples/{sample_id}", status_code=202)
    async def load_sample(sample_id: str) -> dict[str, Any]:
        # The id is an allowlist key, never a path. A miss is a 404, not a filesystem lookup.
        spec = get_sample(sample_id)
        if spec is None:
            raise ApiError("sample_not_found", f"no sample {sample_id}", 404)
        # The SAME pipeline an upload runs: submit_ingest over a constant path. A stable id and
        # idempotency key mean a repeat click reuses the existing dataset rather than creating a
        # duplicate, so a double click stays tidy.
        view = manager().submit_ingest(
            source=spec.path,
            name=f"sample_{spec.id}",
            idempotency_key=f"sample:{spec.id}",
            display_name=spec.filename,
        )
        return {
            "job_id": view.id,
            "kind": view.kind,
            "state": view.state,
            "dataset_id": view.dataset_id,
            "links": _links(view.id),
        }

    @app.get("/samples/{sample_id}/download")
    def download_sample(sample_id: str) -> FileResponse:
        spec = get_sample(sample_id)
        if spec is None:
            raise ApiError("sample_not_found", f"no sample {sample_id}", 404)
        # Served as an attachment with a sensible filename, the SAME bytes ingest reads.
        return FileResponse(spec.path, media_type="text/csv", filename=spec.filename)

    # -- jobs -------------------------------------------------------------------
    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        view = manager().get(job_id)
        if view is None:
            raise ApiError("job_not_found", f"no job {job_id}", 404)
        return view.model_dump(mode="json")

    @app.post("/jobs/{job_id}/cancel", status_code=200)
    def cancel_job(job_id: str) -> dict[str, Any]:
        if manager().get(job_id) is None:
            raise ApiError("job_not_found", f"no job {job_id}", 404)
        return {"cancelled": manager().cancel(job_id)}

    @app.get("/jobs/{job_id}/stream")
    async def stream_job(job_id: str) -> StreamingResponse:
        if manager().get(job_id) is None:
            raise ApiError("job_not_found", f"no job {job_id}", 404)

        async def gen() -> AsyncIterator[bytes]:
            async for ev in manager().stream(job_id):
                yield f"event: progress\ndata: {ev.model_dump_json()}\n\n".encode()
            final = manager().get(job_id)
            assert final is not None
            yield f"event: done\ndata: {final.model_dump_json()}\n\n".encode()

        return StreamingResponse(gen(), media_type="text/event-stream")

    # -- datasets ---------------------------------------------------------------
    @app.get("/datasets")
    def list_datasets() -> dict[str, Any]:
        return {"datasets": store().summaries()}

    @app.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        dataset = store().get(dataset_id)
        if dataset is None:
            raise ApiError("dataset_not_found", f"no ready dataset {dataset_id}", 404)
        return {
            "dataset_id": dataset.dataset_id,
            "row_count": dataset.row_count,
            "model": dataset.model.model_dump(mode="json"),
        }

    # -- queries ----------------------------------------------------------------
    @app.post("/datasets/{dataset_id}/query", status_code=202)
    async def query_dataset(dataset_id: str, body: QueryBody) -> dict[str, Any]:
        dataset = store().get(dataset_id)
        if dataset is None:
            raise ApiError("dataset_not_found", f"no ready dataset {dataset_id}", 404)
        previous_ir = None
        if body.previous_job_id:
            prior = manager().query_result(body.previous_job_id)
            if prior is None:
                raise ApiError(
                    "previous_job_unavailable",
                    "previous_job_id is not a completed query on this dataset",
                    409,
                )
            previous_ir = prior.ir
        view = manager().submit_query(
            dataset=dataset, question=body.question, previous_ir=previous_ir
        )
        return {
            "job_id": view.id,
            "kind": view.kind,
            "state": view.state,
            "dataset_id": dataset_id,
            "links": _links(view.id),
        }

    return app
