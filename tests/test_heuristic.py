"""The deterministic heuristic proposer — the no-LLM fallback.

Proves the semantic layer is not LLM-DEPENDENT: with an empty replay dir (no fixture,
no credentials) an unseen CSV still builds a usable model, marked heuristic; and a real
LLM proposal overrides it. The dev fixtures (when present) must build correct measures
heuristically — no nonsense bindings like revenue = quantity x an id column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from nl_insights.binder import bind
from nl_insights.executor import execute
from nl_insights.ingestion import ingest
from nl_insights.interpreter import QueryIR
from nl_insights.provider import LLMRequest, ReplayProvider
from nl_insights.semantic import build_semantic_model
from nl_insights.semantic.ontology import EntityKind

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_RAW_UCI = _ASSETS / "raw-uci-online-retail.csv"
_ENRICHED = _ASSETS / "nl-insights" / "retail-enriched.csv"


def _synthetic(tmp_path: Path) -> Path:
    rows = ["ord,ts,sku,qty,price,amt,region,is_ret"]
    for o in range(6):
        ts = f"2021-01-{(o % 3) + 1:02d}"  # several orders share a day (real data does)
        for k in range(3):
            qty = -1 if (o == 0 and k == 0) else (k + 1)  # >2 distinct; one return line
            price = 1.5 + k
            amt = round(qty * price, 2)
            region = "ABC"[k]
            is_ret = 1 if qty < 0 else 0
            rows.append(f"O{o},{ts},S{k},{qty},{price},{amt},{region},{is_ret}")
    p = tmp_path / "syn.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def _multi_money(tmp_path: Path) -> Path:
    """A quantity and THREE decimal money columns (a discount, a unit price, a shipping
    fee) whose names alone tell them apart — statistics cannot. None is the product of the
    others, so no stored-amount identity can settle which is the per-unit rate."""
    rows = ["ord,ts,sku,qty,discount,price,shipping,region"]
    for o in range(8):
        ts = f"2021-02-{(o % 4) + 1:02d}"
        for k in range(3):
            qty = k + 1  # >2 distinct positive integers → an unambiguous quantity
            discount = 0.25 + k * 0.5  # decimals, but none == qty × another
            price = 3.5 + k
            shipping = 1.75 + k * 0.5
            region = "ABC"[k]
            rows.append(f"O{o},{ts},S{k},{qty},{discount},{price},{shipping},{region}")
    p = tmp_path / "multi_money.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_heuristic_builds_a_usable_model_without_an_llm(tmp_path: Path) -> None:
    replay = ReplayProvider(tmp_path / "no-fixtures")  # guaranteed miss → heuristic
    result = ingest(_synthetic(tmp_path), data_dir=tmp_path / ".data", name="syn")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, replay)

    # every binding is provenance-marked heuristic — the model is honest about its source
    assert {b.provenance for b in model.bindings} == {"heuristic"}
    # and it is genuinely usable: revenue binds and computes
    net = next(m for m in model.measures if m.name == "net_revenue")
    assert net.available
    v = bind(model, QueryIR(measures=["net_revenue"], group_by=["sku"]))
    assert v.plan is not None
    answer = execute(con, v.plan, model, v.caveats)
    assert answer.rows and answer.rows[0]["net_revenue"] is not None


def test_llm_proposal_overrides_the_heuristic(tmp_path: Path) -> None:
    class LLMStub:
        name = "stub"

        def complete(self, request: LLMRequest) -> dict[str, Any]:
            claims = {
                "ord": ("transaction_key", None),
                "ts": ("event_time", None),
                "sku": ("entity_key", "product"),
                "qty": ("additive_quantity", None),
                "price": ("monetary_rate", None),
                "amt": ("monetary_amount", None),
                "region": ("dimension", None),
                "is_ret": ("flag", None),
            }
            return {
                "claims": [
                    {"column": c, "role": r, "entity": e, "confidence": 0.9, "reasons": ["stub"]}
                    for c, (r, e) in claims.items()
                ]
            }

    result = ingest(_synthetic(tmp_path), data_dir=tmp_path / ".data", name="syn")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, LLMStub())
    assert {b.provenance for b in model.bindings} == {"llm"}
    # the LLM knows sku is a PRODUCT (the heuristic can only say 'other' from stats alone)
    assert "sku" in model.entity_key_columns(EntityKind.PRODUCT)


def test_heuristic_abstains_when_several_money_columns_compete(tmp_path: Path) -> None:
    # Three decimal money columns and no arithmetic identity to settle which is the rate:
    # the proposer must ABSTAIN rather than pick the first one and compute revenue from a
    # discount. net_revenue must not bind, and a revenue question must REFUSE.
    replay = ReplayProvider(tmp_path / "no-fixtures")  # guaranteed miss → heuristic
    result = ingest(_multi_money(tmp_path), data_dir=tmp_path / ".data", name="mm")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, replay)

    from nl_insights.semantic.ontology import Role

    roles = {b.role for b in model.bindings if not b.refuted}
    assert Role.MONETARY_RATE not in roles  # no coin-flip rate
    assert Role.MONETARY_AMOUNT not in roles
    net = next(m for m in model.measures if m.name == "net_revenue")
    assert not net.available  # revenue cannot be computed from an unnamed money column
    verdict = bind(model, QueryIR(measures=["net_revenue"], group_by=["sku"]))
    assert verdict.kind.value == "refuse"
    assert verdict.plan is None


@pytest.mark.skipif(not _RAW_UCI.exists(), reason="fetch raw UCI to run")
def test_heuristic_binds_correct_revenue_on_raw_uci(tmp_path: Path) -> None:
    replay = ReplayProvider(tmp_path / "no-fixtures")
    result = ingest(_RAW_UCI, data_dir=tmp_path / ".data", name="raw")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, replay)
    net = next(m for m in model.measures if m.name == "net_revenue")
    assert net.available
    # revenue must be quantity x unit price — NOT quantity x an id column
    assert net.sql == 'sum("Quantity" * "UnitPrice")'


@pytest.mark.skipif(not _ENRICHED.exists(), reason="optional local fixture, not in the repo")
def test_heuristic_uses_the_verified_stored_amount_on_enriched(tmp_path: Path) -> None:
    replay = ReplayProvider(tmp_path / "no-fixtures")
    result = ingest(_ENRICHED, data_dir=tmp_path / ".data", name="enr")
    con = duckdb.connect(str(result.duckdb_path))
    model = build_semantic_model(con, result.table, result.profile, replay)
    net = next(m for m in model.measures if m.name == "net_revenue")
    assert net.available and net.sql == 'sum("line_revenue")'  # the verified stored amount
    assert {b.provenance for b in model.bindings} == {"heuristic"}
