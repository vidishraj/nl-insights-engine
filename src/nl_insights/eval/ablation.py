"""Metamorphic ablation — the crown move for 'stop inventing answers'.

Take an answerable golden case, programmatically DROP a column its answer depends on,
re-ingest the smaller file, and re-bind the SAME plan. If the verdict flips from answer
to refusal, the refusal is driven by the DATA, not by how the question was phrased —
which is exactly the property the brief names. Because it is generated automatically for
every (case, supporting-column) pair, one small suite yields dozens of refusal proofs.

The proposer is unchanged across the ablation (same claims, or the same heuristic), so
the ONLY thing that varies is the presence of the supporting column. That isolates the
cause: the system refuses because the evidence for the measure is gone.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from pydantic import BaseModel

from ..binder import bind
from ..ingestion import ingest
from ..ingestion.dialect import sniff_dialect
from ..provider import LLMProvider
from ..semantic import build_semantic_model
from .suite import Expected, GoldenCase, classify


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ablate_csv(source: Path, drop_column: str, dest: Path) -> Path:
    """Write a copy of ``source`` with ``drop_column`` removed — a real re-ingest input.

    Read as all-varchar so no retyping happens here (the loader re-infers on ingest),
    and preserve the original delimiter so the ablated file round-trips through the
    same sniffing path.
    """
    dialect = sniff_dialect(source)
    delim = _lit(dialect.delimiter)
    header = "true" if dialect.has_header else "false"
    con = duckdb.connect()
    try:
        read = (
            f"read_csv({_lit(str(source))}, delim={delim}, "
            f"header={header}, all_varchar=true, sample_size=-1)"
        )
        con.execute(
            f"COPY (SELECT * EXCLUDE ({_q(drop_column)}) FROM {read}) "
            f"TO {_lit(str(dest))} (HEADER, DELIMITER {delim})"
        )
    finally:
        con.close()
    return dest


class AblationResult(BaseModel):
    case: str
    dropped_column: str
    refused: bool  # did the verdict correctly flip away from 'answer'?
    # The COARSE outcome (answer / refuse / clarify) - all ablation measures, and all it CAN
    # measure: it binds without executing, so there is no finished answer to earn a fine-grained
    # ANSWERABLE vs ANSWER_WITH_CAVEATS class (that class is finalised only post-execute, by
    # Answer.verdict_kind()). Reporting the bind-time fine-grained class here would put a second,
    # provisional class into a report - exactly the drift we are closing everywhere else.
    outcome: str
    reason: str = ""


def run_ablation(
    source: Path,
    *,
    name: str,
    provider: LLMProvider,
    cases: list[GoldenCase],
    work_dir: Path,
) -> list[AblationResult]:
    """For every answerable case and every column it depends on, drop that column,
    rebuild the model, and check the same plan now refuses."""
    results: list[AblationResult] = []
    data_dir = work_dir / ".data"
    for case in cases:
        if case.expected is not Expected.ANSWER:
            continue
        for col in case.supporting_columns:
            dest = work_dir / f"{name}__{case.name}__drop_{col}.csv"
            ablate_csv(source, col, dest)
            result = ingest(dest, data_dir=data_dir, name=f"{name}_{case.name}_{col}")
            con = duckdb.connect(str(result.duckdb_path))
            try:
                model = build_semantic_model(con, result.table, result.profile, provider)
                verdict = bind(model, case.ir)
            finally:
                con.close()
            results.append(
                AblationResult(
                    case=case.name,
                    dropped_column=col,
                    refused=classify(verdict.kind) is not Expected.ANSWER,
                    outcome=classify(verdict.kind).value,
                    reason=verdict.reason or "",
                )
            )
    return results
