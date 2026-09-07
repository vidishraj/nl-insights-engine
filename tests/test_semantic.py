"""Semantic model: verifiers, discovery, and the LLM-proposes/code-disposes build.

Hermetic tests use a stub provider (the seam makes this trivial) so no fixtures or
credentials are needed; two skipif tests pin the dual-fixture behaviour on the real
files when present.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from nl_insights.ingestion import ingest
from nl_insights.provider import LLMRequest
from nl_insights.semantic import (
    Role,
    SemanticModel,
    build_request,
    build_semantic_model,
    propose_roles,
)
from nl_insights.semantic.evidence import build_evidence

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_RAW_UCI = _ASSETS / "raw-uci-online-retail.csv"
_ENRICHED = _ASSETS / "nl-insights" / "retail-enriched.csv"


class StubProvider:
    """A provider that returns fixed role claims — the injectable seam for tests."""

    name = "stub"

    def __init__(self, claims: dict[str, tuple[str, str | None]]) -> None:
        self._claims = claims

    def complete(self, request: LLMRequest) -> dict[str, object]:
        return {
            "claims": [
                {"column": c, "role": r, "entity": e, "confidence": 0.9, "reasons": ["stub"]}
                for c, (r, e) in self._claims.items()
            ]
        }


def _synthetic(tmp_path: Path) -> Path:
    lines = ["order_id,ts,product,qty,price,amount,country"]
    for o in range(8):
        ts = f"2021-01-{o + 1:02d}"
        for k in range(3):
            qty = -1 if (o < 2 and k == 0) else 2  # first two orders carry a return line
            price = 5.0
            country = "US" if o % 2 else "UK"
            lines.append(f"O{o},{ts},P{k},{qty},{price},{qty * price:.1f},{country}")
    p = tmp_path / "orders.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


_SYNTH_CLAIMS: dict[str, tuple[str, str | None]] = {
    "order_id": ("transaction_key", None),
    "ts": ("event_time", None),
    "product": ("entity_key", "product"),
    "qty": ("additive_quantity", None),
    "price": ("monetary_rate", None),
    "amount": ("monetary_amount", None),
    "country": ("dimension", None),
}


def _build(path: Path, tmp_path: Path, claims: dict[str, tuple[str, str | None]]) -> SemanticModel:
    result = ingest(path, data_dir=tmp_path / ".data")
    con = duckdb.connect(str(result.duckdb_path))
    try:
        return build_semantic_model(con, result.table, result.profile, StubProvider(claims))
    finally:
        con.close()


def test_build_binds_and_verifies_roles(tmp_path: Path) -> None:
    m = _build(_synthetic(tmp_path), tmp_path, _SYNTH_CLAIMS)
    binding = {b.column: b for b in m.bindings}

    # transaction key verified: every multi-line order shares one timestamp.
    key = binding["order_id"]
    assert key.role is Role.TRANSACTION_KEY
    assert any(v.name == "transaction_key_shares_time" and v.passed for v in key.verifiers)

    # stored amount verified row-wise → three roles bound at once.
    amount = binding["amount"]
    assert any(v.name == "stored_amount_identity" and v.passed for v in amount.verifiers)


def test_revenue_uses_verified_stored_amount(tmp_path: Path) -> None:
    m = _build(_synthetic(tmp_path), tmp_path, _SYNTH_CLAIMS)
    net = next(x for x in m.measures if x.name == "net_revenue")
    assert net.available
    assert net.sql == 'sum("amount")'  # the verified stored amount, not the derived product
    order_count = next(x for x in m.measures if x.name == "order_count")
    assert order_count.sql == 'count(DISTINCT "order_id")'


def test_returns_discovered_from_negatives_without_a_flag(tmp_path: Path) -> None:
    m = _build(_synthetic(tmp_path), tmp_path, _SYNTH_CLAIMS)
    # No explicit flag/category/prefix in the synthetic file → the negative quantity IS
    # the return signal.
    assert m.returns.kind == "derived_negative_quantity"


def _text_flag_fixture(tmp_path: Path) -> Path:
    # A two-valued TEXT marker ('charge'/'refund'), not a 0/1 flag — refund rows are
    # negative. Casting such a flag to INTEGER during discovery once crashed the whole
    # ingest; it must instead be found as a category.
    lines = ["ref,ts,product,qty,price,line_kind"]
    for o in range(10):  # enough refund rows to clear the category detector's support floor
        ts = f"2021-01-{o + 1:02d}"
        for k in range(3):
            refund = k == 0  # one refund line per order → 10 refunds, all negative
            qty = -1 if refund else 2
            lines.append(f"R{o},{ts},P{k},{qty},5.0,{'refund' if refund else 'charge'}")
    p = tmp_path / "textflag.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_text_valued_returns_flag_is_a_category_not_an_integer_cast_crash(tmp_path: Path) -> None:
    claims: dict[str, tuple[str, str | None]] = {
        "ref": ("transaction_key", None),
        "ts": ("event_time", None),
        "product": ("entity_key", "product"),
        "qty": ("additive_quantity", None),
        "price": ("monetary_rate", None),
        "line_kind": ("flag", None),  # two-valued text → proposed as a flag
    }
    # The build must SUCCEED (no ConversionException from CAST('refund' AS INTEGER)) and
    # discover the returns convention as a category on the text marker.
    m = _build(_text_flag_fixture(tmp_path), tmp_path, claims)
    assert m.returns.kind == "explicit_category"
    assert m.returns.column == "line_kind" and m.returns.return_values == ["refund"]
    assert any(x.name == "net_revenue" and x.available for x in m.measures)


def test_revenue_is_derived_when_no_stored_amount(tmp_path: Path) -> None:
    # Drop the amount column's claim → revenue must fall back to quantity × rate.
    claims = {k: v for k, v in _SYNTH_CLAIMS.items() if k != "amount"}
    m = _build(_synthetic(tmp_path), tmp_path, claims)
    net = next(x for x in m.measures if x.name == "net_revenue")
    assert net.sql == 'sum("qty" * "price")'


def _labelled_fixture(tmp_path: Path) -> Path:
    """A file that mixes product lines with several non-product line kinds (postage,
    fees, …), where a human-readable description labels the product code. Grounds the
    reviewer's trap: ranking by 'description' must exclude the non-product lines."""
    lines = ["sku,description,line_kind,is_prod,qty,price"]
    for k in range(6):  # 6 products, 3 lines each → descriptions repeat (a real FD)
        for _ in range(3):
            lines.append(f"P{k},DESC {k},PRODUCT,1,2,{k + 1}")
    # ≥5 distinct non-product line kinds so line_kind→is_prod clears the FD support floor.
    for cat, price in [
        ("POSTAGE", 1000),
        ("BANK_FEE", 50),
        ("DISCOUNT", 40),
        ("MANUAL", 30),
        ("CARRIAGE", 20),
    ]:
        for _ in range(3):
            lines.append(f"S_{cat},DESC {cat},{cat},0,2,{price}")
    p = tmp_path / "labelled.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


_LABELLED_CLAIMS: dict[str, tuple[str, str | None]] = {
    "sku": ("entity_key", "product"),
    "description": ("description", None),
    "line_kind": ("dimension", None),
    "is_prod": ("flag", None),
    "qty": ("additive_quantity", None),
    "price": ("monetary_rate", None),
}


def test_description_is_discovered_as_a_product_label(tmp_path: Path) -> None:
    m = _build(_labelled_fixture(tmp_path), tmp_path, _LABELLED_CLAIMS)
    # discovered from the FD, NOT from the name: 'description' labels 'sku' (a product).
    label = next(x for x in m.labels if x.column == "description")
    assert label.entity_column == "sku"
    assert label.entity == "product"
    # so ranking by description is ranking products.
    assert m.product_identifier_columns() == {"sku", "description"}
    # and the partition dimension was discovered with a single fact value.
    pd = next(x for x in m.partition_dimensions if x.column == "line_kind")
    assert pd.fact_value == "PRODUCT"


def test_ranking_by_description_excludes_non_product_lines_end_to_end(tmp_path: Path) -> None:
    from nl_insights.binder import VerdictKind, bind
    from nl_insights.executor import execute
    from nl_insights.interpreter import QueryIR, TopK

    result = ingest(_labelled_fixture(tmp_path), data_dir=tmp_path / ".data")
    con = duckdb.connect(str(result.duckdb_path))
    try:
        m = build_semantic_model(con, result.table, result.profile, StubProvider(_LABELLED_CLAIMS))
        v = bind(
            m,
            QueryIR(
                measures=["net_revenue"],
                group_by=["description"],
                top_k=TopK(measure="net_revenue", k=10),
            ),
        )
        assert v.kind is VerdictKind.ANSWER_WITH_CAVEATS
        assert v.plan is not None
        a = execute(con, v.plan, m, v.caveats)
    finally:
        con.close()
    # postage priced at 1000 would top a naive ranking; the partition removes it.
    assert "line_kind" in a.sql and "PRODUCT" in a.sql
    descriptions = {r["description"] for r in a.rows}
    assert "DESC POSTAGE" not in descriptions
    assert descriptions == {f"DESC {k}" for k in range(6)}  # only the product lines
    assert a.rows[0]["net_revenue"] == 36  # P5: 3 lines × qty 2 × price 6


def test_stored_amount_identity_fails_when_it_does_not_hold(tmp_path: Path) -> None:
    lines = ["order_id,ts,qty,price,amount"]
    for o in range(8):
        for _ in range(2):
            lines.append(f"O{o},2021-01-0{o + 1},2,5.0,999.0")  # amount != qty*price
    p = tmp_path / "bad.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    claims = {
        "order_id": ("transaction_key", None),
        "ts": ("event_time", None),
        "qty": ("additive_quantity", None),
        "price": ("monetary_rate", None),
        "amount": ("monetary_amount", None),
    }
    m = _build(p, tmp_path, claims)
    amount = next(b for b in m.bindings if b.column == "amount")
    assert any(v.name == "stored_amount_identity" and not v.passed for v in amount.verifiers)


def test_proposer_filters_unknown_columns_and_builds_a_schema_request(tmp_path: Path) -> None:
    result = ingest(_synthetic(tmp_path), data_dir=tmp_path / ".data")
    evidence = build_evidence(result.profile)
    request = build_request(evidence, "m")
    assert request.schema["required"] == ["claims"]  # forced structured output

    class Rogue:
        name = "rogue"

        def complete(self, request: LLMRequest) -> dict[str, object]:
            return {
                "claims": [
                    {"column": "does_not_exist", "role": "flag", "confidence": 1.0, "reasons": []}
                ]
            }

    assert propose_roles(Rogue(), evidence, "m") == []  # unknown columns dropped


@pytest.mark.skipif(not _ENRICHED.exists(), reason="optional local fixture, not in the repo")
def test_enriched_verified_amount_partition_and_period(tmp_path: Path) -> None:
    claims = {
        "invoice_no": ("transaction_key", None),
        "invoice_type": ("dimension", None),
        "invoice_date": ("event_time", None),
        "invoice_year": ("dimension", None),
        "invoice_quarter": ("dimension", None),
        "invoice_month": ("dimension", None),
        "invoice_dow": ("dimension", None),
        "invoice_hour": ("dimension", None),
        "is_complete_quarter": ("flag", None),
        "stock_code": ("entity_key", "product"),
        "description": ("description", None),
        "line_type": ("dimension", None),
        "is_product_line": ("flag", None),
        "is_revenue_line": ("flag", None),
        "quantity": ("additive_quantity", None),
        "unit_price": ("monetary_rate", None),
        "line_revenue": ("monetary_amount", None),
        "is_return": ("flag", None),
        "customer_id": ("entity_key", "customer"),
        "is_identified_customer": ("flag", None),
        "country": ("dimension", None),
        "is_country_known": ("flag", None),
        "operator_note": ("ignore", None),
        "has_negative_price": ("flag", None),
        "is_extreme_quantity": ("flag", None),
    }
    m = _build(_ENRICHED, tmp_path, claims)
    amount = next(b for b in m.bindings if b.column == "line_revenue")
    assert any(v.name == "stored_amount_identity" and v.passed for v in amount.verifiers)
    assert m.returns.kind == "explicit_flag" and m.returns.column == "is_return"
    parts = {p.column: p.fact_value for p in m.partition_dimensions}
    assert parts.get("line_type") == "PRODUCT"  # the fact-partition, discovered not hardcoded
    assert any(p.flag_column == "is_complete_quarter" for p in m.period_completeness)
    assert m.coverage["customer_id"] == pytest.approx(0.751, abs=0.01)
    # the transaction key genuinely verifies here → order_count binds (unaffected by .14)
    assert next(x for x in m.measures if x.name == "order_count").available


@pytest.mark.skipif(not _RAW_UCI.exists(), reason="run `make data` to fetch raw UCI")
def test_raw_uci_derives_revenue_and_discovers_c_prefix_returns(tmp_path: Path) -> None:
    claims = {
        "InvoiceNo": ("transaction_key", None),
        "StockCode": ("entity_key", "product"),
        "Description": ("description", None),
        "Quantity": ("additive_quantity", None),
        "InvoiceDate": ("event_time", None),
        "UnitPrice": ("monetary_rate", None),
        "CustomerID": ("entity_key", "customer"),
        "Country": ("dimension", None),
    }
    m = _build(_RAW_UCI, tmp_path, claims)
    net = next(x for x in m.measures if x.name == "net_revenue")
    assert net.sql == 'sum("Quantity" * "UnitPrice")'  # derived — no stored amount exists
    assert m.returns.kind == "derived_key_prefix"
    assert m.returns.return_values == ["C"]  # the cancellation convention, discovered
    assert not m.partition_dimensions  # no flags → nothing to partition on
    assert next(x for x in m.measures if x.name == "order_count").available  # key verifies (0.998)


def test_refuted_transaction_key_is_demoted_and_unbinds_measures(tmp_path: Path) -> None:
    # Rows sharing txn_ref carry DIFFERENT timestamps → the key verifier refutes it.
    # A refuted role must be dropped from resolution and must not bind measures.
    lines = ["txn_ref,sold_on,qty,price"]
    for t in range(20):
        lines.append(f"T{t},2021-01-{(2 * t) % 28 + 1:02d},2,3.5")
        lines.append(f"T{t},2021-01-{(2 * t + 1) % 28 + 1:02d},1,5.0")
    p = tmp_path / "b2b.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    claims = {
        "txn_ref": ("transaction_key", None),
        "sold_on": ("event_time", None),
        "qty": ("additive_quantity", None),
        "price": ("monetary_rate", None),
    }
    m = _build(p, tmp_path, claims)

    key = next(b for b in m.bindings if b.column == "txn_ref")
    assert key.status.value == "refuted"  # SQL disproved the proposal
    assert key.confidence == 0.0  # confidence reflects the disproof, not the 0.9 claim
    assert m.first_in_role(Role.TRANSACTION_KEY) is None  # unreachable by default

    avail = {x.name: x.available for x in m.measures}
    assert not avail["order_count"]  # cannot count orders on a non-key
    assert not avail["basket_cooccurrence"]
    assert avail["net_revenue"]  # derived revenue still holds (quantity × rate verify)
