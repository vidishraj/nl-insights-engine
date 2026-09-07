"""Runnable eval harness: ``python -m nl_insights.eval``.

Prints the anti-hardcoding lint result and a demonstration golden-suite confusion
matrix on a small generated dataset, using the credential-free heuristic proposer so it
runs anywhere with no fixtures. Per-dataset golden suites (which legitimately name real
columns) live under ``tests/`` and run via pytest; this script is the at-a-glance
'how do you know it works' artifact — the matrix and the false-answer rate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb

from ..ingestion import ingest
from ..interpreter import QueryIR, TopK
from ..provider import ReplayProvider
from ..semantic import build_semantic_model
from .lint import main as lint_main
from .suite import Expected, GoldenCase, format_report, run_suite

_COLUMNS = ["order_id", "ts", "product", "qty", "price", "customer"]


def _demo_dataset(path: Path) -> Path:
    rows = [",".join(_COLUMNS)]
    for o in range(8):
        ts = f"2021-01-{(o % 3) + 1:02d}"
        for k in range(3):
            rows.append(f"O{o},{ts},P{k},{2 + k},{1.5 + k},C{o}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _demo_cases() -> list[GoldenCase]:
    return [
        GoldenCase(
            name="top_products_by_revenue",
            question="top products by revenue",
            ir=QueryIR(
                measures=["net_revenue"],
                group_by=["product"],
                top_k=TopK(measure="net_revenue", k=3),
            ),
            expected=Expected.ANSWER,
        ),
        GoldenCase(
            name="unmet_concept_not_in_data",
            question="how much tax did we owe",
            ir=QueryIR(measures=["tax"]),
            expected=Expected.REFUSE,
        ),
        GoldenCase(
            name="revenue_by_unknown_dimension",
            question="revenue by warehouse",
            ir=QueryIR(measures=["net_revenue"], group_by=["warehouse"]),
            expected=Expected.REFUSE,
        ),
    ]


def main() -> int:
    print("== anti-hardcoding lint ==")
    lint_status = lint_main()

    print("\n== demonstration golden suite (heuristic proposer, no credentials) ==")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = ingest(_demo_dataset(tmp_path / "orders.csv"), data_dir=tmp_path / ".data")
        con = duckdb.connect(str(result.duckdb_path))
        model = build_semantic_model(
            con, result.table, result.profile, ReplayProvider(tmp_path / "no-fixtures")
        )
        report = run_suite("demo-orders", model, con, _demo_cases())
        con.close()
    print(format_report(report))

    # non-zero if the lint failed or the suite invented any answer
    return 1 if (lint_status or report.false_answer_rate > 0.0 or not report.passed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
