"""The eval component: golden suites + confusion matrix, metamorphic ablation, header-
stripping generality, a binder-mutation teeth-check, and the anti-hardcoding lint.

All hermetic (a stub provider, tiny synthetic files) so it runs in CI with no fixtures
or credentials and stays disk-cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from nl_insights.eval import (
    Expected,
    GoldenCase,
    run_ablation,
    run_suite,
    scan,
    strip_headers,
)
from nl_insights.eval import suite as suite_mod
from nl_insights.ingestion import ingest
from nl_insights.interpreter import QueryIR, TopK
from nl_insights.provider import LLMRequest
from nl_insights.semantic import build_semantic_model

_COLUMNS = ["order_id", "ts", "product", "qty", "price", "customer"]
_CLAIMS = {
    "order_id": ("transaction_key", None),
    "ts": ("event_time", None),
    "product": ("entity_key", "product"),
    "qty": ("additive_quantity", None),
    "price": ("monetary_rate", None),
    "customer": ("entity_key", "customer"),
}


class StubProvider:
    """Positional stub: maps whatever columns exist (by their known name) to roles, so it
    keeps working after ablation (dropped columns are simply absent from its claims)."""

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


def _write_orders(path: Path) -> Path:
    rows = [",".join(_COLUMNS)]
    for o in range(8):
        ts = f"2021-01-{(o % 3) + 1:02d}"
        for k in range(3):
            rows.append(f"O{o},{ts},P{k},{2 + k},{1.5 + k},C{o}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _cases() -> list[GoldenCase]:
    return [
        GoldenCase(
            name="top_products_by_revenue",
            question="what are the top products by revenue",
            ir=QueryIR(
                measures=["net_revenue"],
                group_by=["product"],
                top_k=TopK(measure="net_revenue", k=3),
            ),
            expected=Expected.ANSWER,
            supporting_columns=["product", "qty", "price"],
        ),
        GoldenCase(
            name="how_many_orders",
            question="how many orders were there",
            ir=QueryIR(measures=["order_count"]),
            expected=Expected.ANSWER,
            supporting_columns=["order_id"],
        ),
        GoldenCase(
            name="profit_is_not_in_the_data",
            question="how much profit did we make",  # presupposition — must refuse
            ir=QueryIR(measures=["profit"]),
            expected=Expected.REFUSE,
        ),
        GoldenCase(
            name="group_by_unknown_dimension",
            question="revenue by region",  # there is no region column
            ir=QueryIR(measures=["net_revenue"], group_by=["region"]),
            expected=Expected.REFUSE,
        ),
        GoldenCase(
            name="followup_that_changed_nothing",
            question="and?",
            ir=QueryIR(
                measures=["net_revenue"],
                notes=["follow-up changed nothing — needs clarification"],
            ),
            expected=Expected.CLARIFY,
        ),
    ]


def _build(path: Path, tmp_path: Path, claims: dict[str, tuple[str, str | None]]):  # type: ignore[no-untyped-def]
    result = ingest(path, data_dir=tmp_path / ".data", name=path.stem)
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, StubProvider(claims))
    return con, model


def test_golden_suite_has_zero_false_answers(tmp_path: Path) -> None:
    con, model = _build(_write_orders(tmp_path / "orders.csv"), tmp_path, _CLAIMS)
    report = run_suite("orders", model, con, _cases())
    assert report.passed  # every case earned its expected verdict class
    assert report.false_answer_rate == 0.0  # nothing invented — THE metric
    assert report.false_refusal_rate == 0.0
    # the confusion matrix is diagonal (2 answers, 2 refusals, 1 clarify)
    assert report.matrix["answer"]["answer"] == 2
    assert report.matrix["refuse"]["refuse"] == 2
    assert report.matrix["clarify"]["clarify"] == 1


def test_metamorphic_ablation_flips_answers_to_refusals(tmp_path: Path) -> None:
    src = _write_orders(tmp_path / "orders.csv")
    results = run_ablation(
        src, name="orders", provider=StubProvider(_CLAIMS), cases=_cases(), work_dir=tmp_path
    )
    # every (answerable case, supporting column) pair must refuse once the column is gone
    assert results  # generated automatically — dozens of proofs for free at scale
    for r in results:
        assert r.refused, f"{r.case} still answered after dropping {r.dropped_column}: {r.outcome}"


def test_binder_mutation_makes_the_suite_go_red(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Disable the binder's judgement (simulate a lost refusal check): every plan now
    # 'answers'. The suite must catch it — false-answer rate jumps and it fails. This is
    # the teeth-check: a green suite means the checks are actually load-bearing.
    from nl_insights.binder import Verdict, VerdictKind

    def always_answer(_model: Any, _ir: Any) -> Verdict:
        return Verdict(kind=VerdictKind.ANSWERABLE, plan=None)

    con, model = _build(_write_orders(tmp_path / "orders.csv"), tmp_path, _CLAIMS)
    monkeypatch.setattr(suite_mod, "bind", always_answer)
    report = run_suite("orders", model, con, _cases())
    assert not report.passed
    assert report.false_answer_rate > 0.0  # the refuse/clarify cases are now invented answers


def test_generality_headers_stripped_still_infers_and_answers(tmp_path: Path) -> None:
    from nl_insights.provider import ReplayCacheMiss
    from nl_insights.semantic.ontology import Role

    src = _write_orders(tmp_path / "orders.csv")
    stripped = strip_headers(src, tmp_path / "stripped.csv")

    # THE point: roles must be INFERRED from the columns' content, not handed in by
    # position. A provider that raises ReplayCacheMiss forces the deterministic heuristic
    # proposer — so nothing is told the answer; the columns are col_1..col_6 with no hint.
    class _ForceHeuristic:
        name = "miss"

        def complete(self, request: LLMRequest) -> dict[str, Any]:
            raise ReplayCacheMiss("force the content-based heuristic proposer")

    result = ingest(stripped, data_dir=tmp_path / ".data", name="stripped")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, _ForceHeuristic())

    # Roles were inferred from CONTENT alone: the date column and the additive quantity are
    # found under anonymous names, and every binding is heuristic-provenance (no LLM, no
    # positional cheat). col_2 was 'ts', col_4 was 'qty' before stripping.
    assert all(b.provenance == "heuristic" for b in model.bindings)
    et = model.first_in_role(Role.EVENT_TIME)
    assert et is not None and et.column == "col_2"
    qty_cols = {b.column for b in model.bindings if b.role == Role.ADDITIVE_QUANTITY}
    assert "col_4" in qty_cols
    # a measure the inferred structure supports is available (units from the quantity)...
    assert any(m.name == "units_sold" and m.available for m in model.measures)

    # ...so the same question, over the anonymous columns, still ANSWERS, and an unknown
    # column still REFUSES — structure, not names, drives both.
    report = run_suite(
        "stripped",
        model,
        con,
        [
            GoldenCase(
                name="units_by_anonymous_product",
                question="units sold by product",
                ir=QueryIR(measures=["units_sold"], group_by=["col_3"]),
                expected=Expected.ANSWER,
            ),
            GoldenCase(
                name="unknown_anonymous_column",
                question="units by col_99",
                ir=QueryIR(measures=["units_sold"], group_by=["col_99"]),
                expected=Expected.REFUSE,
            ),
        ],
    )
    assert report.passed
    assert report.false_answer_rate == 0.0
    con.close()


def test_anti_hardcoding_lint_is_clean() -> None:
    # The engine source must contain no development-dataset identifier. This makes the
    # 'nothing is dataset-specific' claim falsifiable — and green here means it holds.
    violations = scan()
    assert violations == [], "\n".join(f"{v.path}:{v.line} {v.token}" for v in violations)
