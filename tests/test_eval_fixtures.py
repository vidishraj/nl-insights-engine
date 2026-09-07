"""Real-fixture golden checks: the same question, different CORRECT answers.

'Top products by revenue' legitimately differs across the two development files, and
that difference is the thesis in one example:
  - ENRICHED carries a line-type column, so a fact partition is discovered and postage
    is EXCLUDED (pinned in test_executor::…excludes_non_product_lines and …matches_
    ground_truth).
  - RAW UCI has no such column, so there is no evidence to exclude anything and postage
    is correctly INCLUDED (pinned here).
The system acts on evidence, not on names, and refuses to invent structure that a file
does not carry. Skipped unless the fixtures are present; each ingests one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from nl_insights.binder import VerdictKind, bind
from nl_insights.eval import Expected, GoldenCase, run_suite
from nl_insights.executor import execute
from nl_insights.ingestion import ingest
from nl_insights.interpreter import QueryIR, TopK
from nl_insights.provider import LLMRequest
from nl_insights.semantic import build_semantic_model

_RAW_UCI = Path(__file__).resolve().parents[1] / "assets" / "raw-uci-online-retail.csv"

_RAW_CLAIMS = {
    "InvoiceNo": ("transaction_key", None),
    "StockCode": ("entity_key", "product"),
    "Description": ("description", None),
    "Quantity": ("additive_quantity", None),
    "InvoiceDate": ("event_time", None),
    "UnitPrice": ("monetary_rate", None),
    "CustomerID": ("entity_key", "customer"),
    "Country": ("dimension", None),
}


class StubProvider:
    name = "stub"

    def __init__(self, claims: dict[str, tuple[str, str | None]]) -> None:
        self._claims = claims

    def complete(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "claims": [
                {"column": c, "role": r, "entity": e, "confidence": 0.9, "reasons": ["stub"]}
                for c, (r, e) in self._claims.items()
            ]
        }


@pytest.mark.skipif(not _RAW_UCI.exists(), reason="fetch raw UCI to run")
def test_raw_uci_includes_non_product_lines_because_no_evidence_excludes_them(
    tmp_path: Path,
) -> None:
    result = ingest(_RAW_UCI, data_dir=tmp_path / ".data", name="raw")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, StubProvider(_RAW_CLAIMS))

    # no line-type column exists → NO fact partition is discovered → nothing to exclude.
    assert not any(p.fact_value for p in model.partition_dimensions)

    # 'top products by revenue' answers, applies NO partition, and postage is present —
    # the opposite of the enriched file, and correct here because the evidence is absent.
    ir = QueryIR(
        measures=["net_revenue"],
        group_by=["StockCode"],
        top_k=TopK(measure="net_revenue", k=50),
    )
    verdict = bind(model, ir)
    assert verdict.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}
    assert not verdict.plan.applied_filters  # no product-line partition applied
    answer = execute(con, verdict.plan, model, verdict.caveats)
    assert "POST" in {r["StockCode"] for r in answer.rows}  # the postage line is included

    # and it is still a real, non-inventing suite: the presupposition question refuses.
    report = run_suite(
        "raw-uci",
        model,
        con,
        [
            GoldenCase(
                name="top_products_by_revenue",
                question="top products by revenue",
                ir=ir,
                expected=Expected.ANSWER,
            ),
            GoldenCase(
                name="profit_presupposition",
                question="how much profit did we make",
                ir=QueryIR(measures=["profit"]),
                expected=Expected.REFUSE,
            ),
        ],
    )
    con.close()
    assert report.passed and report.false_answer_rate == 0.0
