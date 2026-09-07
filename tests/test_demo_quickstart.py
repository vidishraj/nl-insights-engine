"""The zero-credential quickstart genuinely answers — and refuses — on the default path.

Drives the COMMITTED demo dataset through the REAL replay provider (the bundled
fixtures, no key), exactly as `make run` does. The cassettes are REAL recordings from a
live model — both role proposal and interpretation — so this exercises the full 'LLM
proposes, code disposes' pipeline hermetically (the recorded proposal even types
`product`/`customer` as entities, which the heuristic cannot). If a fixture ever stops
matching (a key drift), this fails loudly rather than the quickstart silently regressing
to `needs_llm`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from nl_insights.binder import VerdictKind, bind
from nl_insights.config import Settings
from nl_insights.executor import execute
from nl_insights.ingestion import ingest
from nl_insights.interpreter import interpret
from nl_insights.provider import ReplayProvider
from nl_insights.semantic import build_semantic_model

_ROOT = Path(__file__).resolve().parents[1]
_DEMO = _ROOT / "assets" / "demo" / "store-orders.csv"


def _demo_model(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = Settings()
    provider = ReplayProvider(settings.fixtures_dir)  # the committed bundled fixtures
    result = ingest(_DEMO, data_dir=tmp_path / ".data", name="demo")
    con = duckdb.connect(str(result.duckdb_path))
    # role proposal hits the committed cassette → an LLM-produced model, exactly like the
    # server on the default replay path (the tests below assert provenance == 'llm')
    model = build_semantic_model(con, result.table, result.profile, provider)
    return con, model, provider, settings.model


def test_proposal_key_is_stable_and_name_independent(tmp_path: Path) -> None:
    # The replay cache key for role proposal must be a pure function of the file's
    # STRUCTURE — identical bytes yield one key across repeat ingests (guards the
    # profiler's ORDER BY determinism) and across upload names (guards excluding
    # dataset_id). Otherwise the zero-credential quickstart only works for one exact name.
    from nl_insights.semantic.evidence import build_evidence
    from nl_insights.semantic.proposer import build_request

    settings = Settings()
    keys = set()
    for i, name in enumerate(["demo", "store-orders", "retail", "demo", "my-upload"]):
        result = ingest(_DEMO, data_dir=tmp_path / f".data{i}", name=name)
        keys.add(build_request(build_evidence(result.profile), settings.model).cache_key())
    assert len(keys) == 1, f"identical bytes produced {len(keys)} proposal keys (want 1)"


@pytest.mark.parametrize(
    "question,expect_answer",
    [
        ("top products by revenue", True),
        ("how many orders were there", True),
        ("units sold by product", True),
        # a genuine live refusal — nothing in the file maps to a supplier count, and the
        # recorded model declined rather than inventing one.
        ("how many distinct suppliers are there", False),
    ],
)
def test_bundled_questions_resolve_on_the_default_replay_path(
    tmp_path: Path, question: str, expect_answer: bool
) -> None:
    con, model, provider, llm_model = _demo_model(tmp_path)
    # the recorded proposal makes this an LLM-typed model, not the heuristic fallback
    assert {b.provenance for b in model.bindings} == {"llm"}
    ir = interpret(provider, model, question, llm_model=llm_model)  # replay hit, no key
    verdict = bind(model, ir)
    if expect_answer:
        assert verdict.kind in {VerdictKind.ANSWERABLE, VerdictKind.ANSWER_WITH_CAVEATS}
        answer = execute(con, verdict.plan, model, verdict.caveats)
        assert answer.rows  # a real answer computed by SQL over the real data
    else:
        assert verdict.kind is VerdictKind.REFUSE
    con.close()
