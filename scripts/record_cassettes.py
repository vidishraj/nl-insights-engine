"""Record REAL cassettes from the live ambient model — role proposal AND interpretation —
and report the LLM-vs-heuristic model diff on both datasets.

This is what makes the semantic model actually LLM-produced on the committed path: the
demo's role proposal and question interpretations are recorded from a live model (not
hand-authored), so `make run` replays the *real* LLM pipeline with zero credentials. The
recorded plans are still binder-validated and executed by real SQL — replay serves what
the model said, the deterministic layers dispose of it exactly as in production.

    uv run --extra ambient python scripts/record_cassettes.py

Re-run to refresh after a prompt/schema change. Requires the ambient path (the local
`claude` CLI); it makes live calls.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import duckdb

from nl_insights.binder import bind
from nl_insights.config import Settings
from nl_insights.executor import execute
from nl_insights.ingestion import ingest
from nl_insights.interpreter import interpret
from nl_insights.provider import RecordingProvider, ReplayProvider
from nl_insights.provider.ambient import AmbientProvider
from nl_insights.semantic import build_semantic_model

_ROOT = Path(__file__).resolve().parents[1]
_FIX = _ROOT / "fixtures" / "llm"
_DEMO = _ROOT / "assets" / "demo" / "store-orders.csv"
_RAW = _ROOT / "assets" / "raw-uci-online-retail.csv"


def _clean_cassettes() -> None:
    # Remove every cassette except the seam smoke fixture, so a re-record leaves only
    # real recordings (no stale authored/orphaned keys).
    for path in _FIX.glob("*.json"):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("recorded_by") != "canned":
            path.unlink()


# The bundled demo questions — three clean answers, one capability only the LLM's entity
# typing unlocks (basket), and one GENUINE live refusal (nothing maps to a supplier count).
_DEMO_QUESTIONS = [
    "top products by revenue",
    "how many orders were there",
    "units sold by product",
    "which products are most often bought together",
    "how many distinct suppliers are there",
]


def _summary(model) -> dict[str, tuple]:  # type: ignore[no-untyped-def]
    return {
        b.column: (b.role.value, b.entity.value if b.entity else None, round(b.confidence, 2))
        for b in model.bindings
    }


def _print_diff(label: str, heuristic, llm) -> None:  # type: ignore[no-untyped-def]
    h, m = _summary(heuristic), _summary(llm)
    print(f"\n=== {label}: LLM vs heuristic model ===")
    print(f"  heuristic measures: {[x.name for x in heuristic.measures if x.available]}")
    print(f"  LLM       measures: {[x.name for x in llm.measures if x.available]}")
    for col in sorted(set(h) | set(m)):
        if h.get(col) != m.get(col):
            print(f"  {col:14} heuristic={h.get(col)}  LLM={m.get(col)}")


def main() -> int:
    settings = Settings()
    live = AmbientProvider()
    recorder = RecordingProvider(live, _FIX)

    _clean_cassettes()

    # --- DEMO: record role proposal + each interpretation, verify, diff ---
    with tempfile.TemporaryDirectory() as tmp:
        result = ingest(_DEMO, data_dir=Path(tmp) / ".data", name="demo")
        con = duckdb.connect(str(result.duckdb_path))
        heuristic = build_semantic_model(
            con, result.table, result.profile, ReplayProvider(Path(tmp) / "none")
        )
        con2 = duckdb.connect(str(result.duckdb_path))
        llm = build_semantic_model(con2, result.table, result.profile, recorder)  # records proposal
        _print_diff("DEMO", heuristic, llm)
        print("\n=== DEMO: recorded interpretations ===")
        for q in _DEMO_QUESTIONS:
            ir = interpret(recorder, llm, q, llm_model=settings.model)  # records interpret
            verdict = bind(llm, ir)
            extra = ""
            if verdict.plan is not None:
                ans = execute(con2, verdict.plan, llm, verdict.caveats)
                extra = f" rows={len(ans.rows)}"
            print(f"  {q!r:48} -> {verdict.kind.value}{extra}")

    # --- RAW UCI: record role proposal + diff (documentary; the file is not committed) ---
    if _RAW.exists():
        with tempfile.TemporaryDirectory() as tmp:
            result = ingest(_RAW, data_dir=Path(tmp) / ".data", name="raw")
            con = duckdb.connect(str(result.duckdb_path))
            heuristic = build_semantic_model(
                con, result.table, result.profile, ReplayProvider(Path(tmp) / "none")
            )
            con2 = duckdb.connect(str(result.duckdb_path))
            llm = build_semantic_model(con2, result.table, result.profile, recorder)
            _print_diff("RAW UCI", heuristic, llm)
    else:
        print("\n(raw UCI not present — skipping its proposal cassette)")

    print(f"\ncassettes now in {_FIX}: {len(list(_FIX.glob('*.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
