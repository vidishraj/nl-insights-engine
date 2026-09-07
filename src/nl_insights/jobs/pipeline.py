"""The synchronous work each job runs — ingest a file, or answer a question.

These are plain blocking functions (DuckDB and the profiler are CPU/IO bound); the job
manager runs them in a worker thread so the event loop stays free and datasets ingest
concurrently. Each takes an ``emit(stage, message)`` callback that narrates progress
AND may raise :class:`Cancelled` at a stage boundary, so cancellation is cooperative
and lands at a safe point rather than tearing a DuckDB call in half.

Ingest is crash-safe by construction: it builds into a STAGING file and only promotes
it (atomic rename) once the whole model is built, so a job that dies mid-ingest never
leaves a half-loaded dataset queryable. The query path preserves THE invariant — it
goes interpret -> bind -> execute, and nothing reaches SQL without passing the binder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..binder import VerdictKind, bind
from ..executor import Answer, NeedsClarification, execute
from ..ingestion import ingest
from ..ingestion.dialect import caller_label
from ..ingestion.loader import dataset_id_for
from ..interpreter import interpret, interpret_followup, merge_ir
from ..interpreter.ir import QueryIR
from ..provider import LLMProvider, ReplayCacheMiss
from ..semantic import build_semantic_model
from .events import JobError
from .store import Dataset, promote, write_sidecar

_log = logging.getLogger(__name__)


class Cancelled(Exception):
    """Raised by an ``emit`` callback when the job was asked to cancel."""


class PipelineError(Exception):
    """A failure with a stable client-facing code (never a stack trace to the caller)."""

    def __init__(self, code: str, message: str, **details: str) -> None:
        # A shaped error must NEVER be empty: some exceptions stringify to '' (a bare
        # TimeoutError/CancelledError has no args), which would hand the client a code and
        # a blank. Floor the message to the code so the surface always says something.
        message = message or code
        super().__init__(message)
        self.error = JobError(code=code, message=message, details=details)


EmitFn = Any  # Callable[[str, str], None]; kept loose to avoid importing the manager


def _scrub_internal_paths(text: str, label: str, *paths: Path) -> str:
    """Belt-and-braces: strip any known server path (the upload, the staging and data dirs)
    from a message about to reach the caller, substituting the user's filename. The ingestion
    layer already names the user's file rather than the path, but a DuckDB or provider error
    raised deeper (e.g. while building the semantic model) can still embed one; the caller must
    never see the data-store layout, so we remove the exact strings we control."""
    for p in paths:
        text = text.replace(str(p), label)
    return text


def run_ingest(
    *,
    source: Path,
    name: str | None,
    provider: LLMProvider,
    model: str,
    data_dir: Path,
    staging_dir: Path,
    emit: EmitFn,
    display_name: str | None = None,
) -> Dataset:
    dataset_id = dataset_id_for(source, name)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / f"{dataset_id}.duckdb"
    canonical_path = data_dir / f"{dataset_id}.duckdb"
    try:
        # Build into the staging dir (same filesystem as data_dir → atomic promote).
        result = ingest(
            source, data_dir=staging_dir, name=name, on_stage=emit, display_name=display_name
        )
        import duckdb

        con = duckdb.connect(str(result.duckdb_path))
        try:
            # Pass the ingester's resolved date orders so the temporal verifier can probe
            # a text date with the RIGHT parser and disclose the interpretation it used.
            semantic = build_semantic_model(
                con,
                result.table,
                result.profile,
                provider,
                model=model,
                on_stage=emit,
                date_orders=result.date_orders,
            )
        finally:
            con.close()

        emit("promoting", "publishing the dataset atomically")
        promote(result.duckdb_path, canonical_path)
        dataset = Dataset(
            dataset_id=dataset_id,
            table=result.table,
            duckdb_path=str(canonical_path),
            row_count=result.profile.row_count,
            model=semantic,
        )
        # Persist the understanding beside the data so a restart rehydrates it, rather than
        # dropping the dataset with the process. DuckDB is the data of record; the sidecar
        # JSON is the understanding of record. A sidecar write that fails AFTER promotion must
        # not fail an otherwise-good ingest — the dataset is usable this session; it just will
        # not survive a restart (a re-ingest persists it), which is strictly better than
        # discarding a fully-built dataset.
        try:
            write_sidecar(dataset)
        except OSError as exc:
            _log.warning(
                "dataset %s ingested but not persisted (sidecar write failed: %s); "
                "it will not survive a restart",
                dataset_id,
                exc,
            )
        return dataset
    except Cancelled:
        raise
    except PipelineError:
        raise
    except ReplayCacheMiss as exc:
        # Understanding an UNSEEN file needs the model; in replay mode with no recorded response
        # say so plainly rather than forward the miss, whose text names the fixtures PATH.
        raise PipelineError(
            "needs_llm",
            "understanding this file needs the model and no recorded response exists in replay "
            "mode; run the server with --provider ambient or --provider apikey to ingest a new "
            "file, or use one of the bundled datasets",
            dataset_id=dataset_id,
        ) from exc
    except Exception as exc:  # ingestion/type/date errors → a shaped, code-bearing error
        # The full detail WITH the real path goes to the operator log (they need it to locate
        # the file); the caller gets a message scrubbed of every server path. str(exc) is '' for
        # a bare TimeoutError/CancelledError; fall back to the class name so it is never blank.
        _log.warning("ingest of dataset %s from %s failed: %s", dataset_id, source, exc)
        label = caller_label(display_name)
        detail = str(exc) or exc.__class__.__name__
        detail = _scrub_internal_paths(detail, label, source, staging_dir, data_dir)
        raise PipelineError("ingest_failed", detail, dataset_id=dataset_id) from exc
    finally:
        # A staging file only survives if we did NOT promote (crash/cancel/failure);
        # remove it so a dead job never leaves a half-built file lying around.
        if staging_path.exists():
            staging_path.unlink()


class QueryResult(BaseModel):
    """The outcome of a query job — a verdict, and an Answer when one was produced.

    A REFUSE/CLARIFY is a SUCCESSFUL job (the system produced the right, honest verdict);
    it is not a failure. ``ir`` is the resolved plan, kept so a follow-up can diff it.
    """

    dataset_id: str
    kind: VerdictKind
    answer: Answer | None = None
    ir: QueryIR
    reason: str | None = None
    evidence: dict[str, Any] = {}
    clarify_question: str | None = None
    options: list[str] = []


def run_query(
    *,
    dataset: Dataset,
    question: str,
    previous_ir: QueryIR | None,
    provider: LLMProvider,
    model: str,
    emit: EmitFn,
) -> QueryResult:
    try:
        emit("interpreting", "translating the question into a typed plan")
        if previous_ir is not None:
            partial = interpret_followup(
                provider, dataset.model, previous_ir, question, llm_model=model
            )
            ir = merge_ir(previous_ir, partial)
        else:
            ir = interpret(provider, dataset.model, question, llm_model=model)

        emit("binding", "validating the plan against the semantic model")
        verdict = bind(dataset.model, ir)  # THE invariant: no SQL without this

        if verdict.kind in {VerdictKind.REFUSE, VerdictKind.CLARIFY}:
            return QueryResult(
                dataset_id=dataset.dataset_id,
                kind=verdict.kind,
                ir=ir,
                reason=verdict.reason,
                evidence=verdict.evidence,
                clarify_question=verdict.clarify_question,
                options=verdict.options,
            )

        assert verdict.plan is not None
        emit("executing", "running the bound plan on DuckDB")
        con = dataset.connect()
        try:
            answer: Answer = execute(con, verdict.plan, dataset.model, verdict.caveats)
        except NeedsClarification as clarify:
            # A data-dependent clarification (e.g. a growth period that is not in the data).
            # The binder validated the plan's structure; this is the one refusal whose
            # evidence only exists once we look at the data. Surface it as a CLARIFY, never
            # a silent zero/NULL over a period that isn't there.
            return QueryResult(
                dataset_id=dataset.dataset_id,
                kind=VerdictKind.CLARIFY,
                ir=ir,
                clarify_question=clarify.question,
                options=clarify.options,
                evidence=clarify.evidence,
            )
        finally:
            con.close()
        # The verdict CLASS is finalised from the FINISHED answer, never the bind-time kind: the
        # answer carries caveats (e.g. a non-fact partition disclosure) that only exist after
        # execute, and Answer.verdict_kind() is the one place that class is decided. Every consumer
        # of this job reads QueryResult.kind, so this is the single class the payload, the SSE done
        # event and the API all report.
        return QueryResult(
            dataset_id=dataset.dataset_id, kind=answer.verdict_kind(), answer=answer, ir=ir
        )
    except Cancelled:
        raise
    except ReplayCacheMiss as exc:
        # Interpreting an UNSEEN question genuinely needs the LLM (unlike ingestion,
        # which falls back to heuristics). Say so plainly instead of leaking a cache key.
        raise PipelineError(
            "needs_llm",
            "this question has no recorded response in replay mode; run the server with "
            "--provider ambient or --provider apikey to interpret new questions live, or "
            "ask one of the bundled example questions",
            dataset_id=dataset.dataset_id,
        ) from exc
    except Exception as exc:
        # Do NOT forward the raw exception: a DuckDB error carries SQL fragments and
        # candidate column names. The binder is supposed to have caught unanswerable
        # plans already, so reaching here is an internal fault — report it as one.
        raise PipelineError(
            "query_failed",
            "the query could not be executed against this dataset",
            dataset_id=dataset.dataset_id,
        ) from exc
