"""Jobs: the concurrency and partial-failure behaviour the brief asks us to defend.

Driven through ``asyncio.run`` (no pytest-asyncio dependency). A combined stub provider
answers both LLM call sites — role claims for ingest, a fixed Query IR for queries — so
the whole path is hermetic: no fixtures, no credentials.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

from nl_insights.ingestion.loader import dataset_id_for
from nl_insights.jobs import DatasetStore, JobManager, JobState
from nl_insights.provider import LLMRequest, ReplayProvider

_ORDERS = "order_id,ts,product,qty,price,customer\n" + "".join(
    f"O{o},2021-01-{o + 1:02d},P{k},{2 + k},{1.5 + k},C{o}\n" for o in range(8) for k in range(3)
)

_CLAIMS = {
    "order_id": ("transaction_key", None),
    "ts": ("event_time", None),
    "product": ("entity_key", "product"),
    "qty": ("additive_quantity", None),
    "price": ("monetary_rate", None),
    "customer": ("entity_key", "customer"),
}
_IR = {
    "measures": ["net_revenue"],
    "group_by": ["product"],
    "top_k": {"measure": "net_revenue", "k": 3},
}


class StubProvider:
    """Answers role-claim requests and query-IR requests off the same seam."""

    name = "stub"

    def complete(self, request: LLMRequest) -> dict[str, Any]:
        props = (request.schema or {}).get("properties", {})
        if "claims" in props:
            return {
                "claims": [
                    {"column": c, "role": r, "entity": e, "confidence": 0.9, "reasons": ["stub"]}
                    for c, (r, e) in _CLAIMS.items()
                ]
            }
        return dict(_IR)


def _write(tmp_path: Path, text: str = _ORDERS, name: str = "orders.csv") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _manager(tmp_path: Path, provider: Any | None = None) -> tuple[JobManager, DatasetStore]:
    store = DatasetStore()
    mgr = JobManager(
        provider=provider or StubProvider(), model="stub-model", data_dir=tmp_path, store=store
    )
    return mgr, store


def test_ingest_emits_stages_and_registers_dataset(tmp_path: Path) -> None:
    async def scenario() -> None:
        mgr, store = _manager(tmp_path)
        src = _write(tmp_path)
        view = mgr.submit_ingest(source=src, name="orders", idempotency_key="k1")
        final = await mgr.wait(view.id)
        assert final.state is JobState.SUCCEEDED
        stages = [e.stage for e in final.events]
        # the semantic model assembling, narrated end-to-end (the demo):
        for expected in ["sniffing", "loading", "profiling", "inferring", "verifying", "promoting"]:
            assert expected in stages
        ds = store.get(dataset_id_for(src, "orders"))
        assert ds is not None and ds.row_count == 24

    asyncio.run(scenario())


def test_ingest_is_idempotent_under_retry(tmp_path: Path) -> None:
    async def scenario() -> None:
        mgr, _ = _manager(tmp_path)
        src = _write(tmp_path)
        a = mgr.submit_ingest(source=src, name="orders", idempotency_key="same")
        b = mgr.submit_ingest(source=src, name="orders", idempotency_key="same")
        assert a.id == b.id  # the retry reused the job, it did not ingest twice
        await mgr.wait(a.id)

    asyncio.run(scenario())


def test_failed_ingest_leaves_no_queryable_dataset(tmp_path: Path) -> None:
    class Boom(StubProvider):
        def complete(self, request: LLMRequest) -> dict[str, Any]:
            raise RuntimeError("provider exploded mid-build")

    async def scenario() -> None:
        mgr, store = _manager(tmp_path, provider=Boom())
        src = _write(tmp_path)
        view = mgr.submit_ingest(source=src, name="orders", idempotency_key="k")
        final = await mgr.wait(view.id)
        assert final.state is JobState.FAILED
        assert final.error is not None and final.error.code == "ingest_failed"
        # shaped: a single-line message, never a multi-line stack trace to the caller.
        assert "\n" not in final.error.message and "Traceback" not in final.error.message
        dataset_id = dataset_id_for(src, "orders")
        assert store.get(dataset_id) is None  # never registered
        assert not (tmp_path / f"{dataset_id}.duckdb").exists()  # never promoted
        assert not (tmp_path / ".staging" / f"{dataset_id}.duckdb").exists()  # staging cleaned

    asyncio.run(scenario())


def test_shaped_error_is_never_empty_even_for_a_bare_exception(tmp_path: Path) -> None:
    # A bare TimeoutError stringifies to '' — the exact trap that handed the live client a
    # code and a blank message. The shaped error must still be human-readable.
    class Hang(StubProvider):
        def complete(self, request: LLMRequest) -> dict[str, Any]:
            raise TimeoutError()  # str(TimeoutError()) == ''

    async def scenario() -> None:
        mgr, _ = _manager(tmp_path, provider=Hang())
        src = _write(tmp_path)
        view = mgr.submit_ingest(source=src, name="orders", idempotency_key="k")
        final = await mgr.wait(view.id)
        assert final.state is JobState.FAILED
        assert final.error is not None and final.error.code == "ingest_failed"
        assert final.error.message.strip()  # NEVER empty — the whole point
        assert "\n" not in final.error.message

    asyncio.run(scenario())


def test_pipeline_error_floors_an_empty_message_to_the_code() -> None:
    from nl_insights.jobs.pipeline import PipelineError

    e = PipelineError("storage_full", "")  # a caller (or bare exc) with no message
    assert e.error.code == "storage_full"
    assert e.error.message == "storage_full"  # floored, never blank


def test_persisted_dataset_rehydrates_and_an_unknown_version_is_refused(tmp_path: Path) -> None:
    import json

    from nl_insights.jobs.store import CURRENT_MODEL_VERSION, load_persisted

    async def scenario() -> None:
        mgr, _ = _manager(tmp_path)
        src = _write(tmp_path)
        view = mgr.submit_ingest(source=src, name="orders", idempotency_key="k")
        await mgr.wait(view.id)

    asyncio.run(scenario())
    ds_id = dataset_id_for(_write(tmp_path), "orders")

    # a fresh store (a "restart") rehydrates the persisted dataset from its sidecar
    fresh = DatasetStore()
    assert load_persisted(fresh, tmp_path) == 1
    assert fresh.get(ds_id) is not None and fresh.get(ds_id).row_count == 24

    # tamper the sidecar to an unknown future version → it is REFUSED, not trusted
    sidecar = next(iter(tmp_path.glob("*.dataset.json")))
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    raw["model"]["version"] = CURRENT_MODEL_VERSION + 1
    sidecar.write_text(json.dumps(raw), encoding="utf-8")
    refused = DatasetStore()
    assert load_persisted(refused, tmp_path) == 0
    assert refused.get(ds_id) is None


def test_older_version_sidecar_is_migrated_by_rederiving_not_left_degraded(tmp_path: Path) -> None:
    # A dataset ingested BEFORE a binder-relied-upon field existed (v1, no categorical_values)
    # must not load silently degraded. It is migrated forward by RE-DERIVING the field from the
    # on-disk DuckDB — the user loses nothing and never re-uploads. This is the Retail_check case.
    import json

    from nl_insights.jobs.store import CURRENT_MODEL_VERSION, load_persisted

    async def scenario() -> None:
        mgr, _ = _manager(tmp_path)
        view = mgr.submit_ingest(source=_write(tmp_path), name="orders", idempotency_key="k")
        await mgr.wait(view.id)

    asyncio.run(scenario())
    ds_id = dataset_id_for(_write(tmp_path), "orders")

    # downgrade the sidecar to the pre-fix schema: version 1, no categorical_values
    sidecar = next(iter(tmp_path.glob("*.dataset.json")))
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    raw["model"]["version"] = 1
    raw["model"].pop("categorical_values", None)
    sidecar.write_text(json.dumps(raw), encoding="utf-8")

    fresh = DatasetStore()
    assert load_persisted(fresh, tmp_path) == 1  # migrated, NOT refused
    ds = fresh.get(ds_id)
    assert ds is not None
    assert ds.model.categorical_values  # re-derived from the data, not left empty
    assert ds.model.version == CURRENT_MODEL_VERSION  # upgraded in memory
    # and the sidecar was upgraded on disk, so it is not re-derived every restart
    upgraded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert upgraded["model"]["version"] == CURRENT_MODEL_VERSION
    assert upgraded["model"]["categorical_values"]


def test_v2_sidecar_migrates_by_rederiving_numeric_columns(tmp_path: Path) -> None:
    # The v3 case, end to end (the Retail_check scenario the lead asked to prove a second time):
    # a dataset ingested at v2 has categorical_values but NO numeric_columns. On load it must
    # migrate in place — re-deriving numeric_columns from the DuckDB column TYPES (no LLM, no
    # re-upload) — and then still ANSWER a generic aggregation, which depends on that field.
    import json

    from nl_insights.binder import VerdictKind, bind
    from nl_insights.interpreter import QueryIR
    from nl_insights.interpreter.ir import Aggregation
    from nl_insights.jobs.store import CURRENT_MODEL_VERSION, load_persisted

    async def scenario() -> None:
        mgr, _ = _manager(tmp_path)
        view = mgr.submit_ingest(source=_write(tmp_path), name="orders", idempotency_key="k")
        await mgr.wait(view.id)

    asyncio.run(scenario())
    ds_id = dataset_id_for(_write(tmp_path), "orders")

    # downgrade to the v2 schema: keep categorical_values, drop numeric_columns, set version 2
    sidecar = next(iter(tmp_path.glob("*.dataset.json")))
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    raw["model"]["version"] = 2
    raw["model"].pop("numeric_columns", None)
    sidecar.write_text(json.dumps(raw), encoding="utf-8")

    fresh = DatasetStore()
    assert load_persisted(fresh, tmp_path) == 1  # migrated forward, NOT refused
    ds = fresh.get(ds_id)
    assert ds is not None
    assert ds.model.numeric_columns  # re-derived from the DuckDB types, not left empty
    assert ds.model.version == CURRENT_MODEL_VERSION
    # and the migrated model still ANSWERS a generic aggregation over a re-derived numeric
    numeric = ds.model.numeric_columns[0]
    con = ds.connect()
    try:
        v = bind(ds.model, QueryIR(aggregations=[Aggregation(func="avg", column=numeric)]))
        assert v.kind is not VerdictKind.REFUSE and v.plan is not None
    finally:
        con.close()
    # the sidecar was upgraded on disk too
    upgraded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert upgraded["model"]["version"] == CURRENT_MODEL_VERSION
    assert upgraded["model"]["numeric_columns"]


def test_rehydration_skips_a_sidecar_whose_data_file_is_gone(tmp_path: Path) -> None:
    from nl_insights.jobs.store import load_persisted

    async def scenario() -> None:
        mgr, _ = _manager(tmp_path)
        view = mgr.submit_ingest(source=_write(tmp_path), name="orders", idempotency_key="k")
        await mgr.wait(view.id)

    asyncio.run(scenario())
    # remove the DuckDB file but leave the sidecar → nothing to query, so skip it
    ds_id = dataset_id_for(_write(tmp_path), "orders")
    (tmp_path / f"{ds_id}.duckdb").unlink()
    fresh = DatasetStore()
    assert load_persisted(fresh, tmp_path) == 0
    assert fresh.get(ds_id) is None


def test_rehydration_refuses_a_sidecar_whose_table_was_swapped_underneath(tmp_path: Path) -> None:
    # Existence is not identity: if the DuckDB table is rebuilt with a different schema/size
    # under a good sidecar, the model would describe columns that no longer exist. Rehydration
    # must REFUSE it rather than silently trust a model that no longer matches the data.
    import duckdb

    from nl_insights.jobs.store import load_persisted

    async def scenario() -> None:
        mgr, store = _manager(tmp_path)
        view = mgr.submit_ingest(source=_write(tmp_path), name="orders", idempotency_key="k")
        await mgr.wait(view.id)
        return store.get(dataset_id_for(_write(tmp_path), "orders"))

    ds = asyncio.run(scenario())
    assert ds is not None
    # swap the table underneath: wholly different columns and a different row count
    con = duckdb.connect(ds.duckdb_path)
    con.execute(f'DROP TABLE "{ds.table}"')
    con.execute(f'CREATE TABLE "{ds.table}" AS SELECT 1 AS unrelated_a, 2 AS unrelated_b')
    con.close()

    fresh = DatasetStore()
    assert load_persisted(fresh, tmp_path) == 0  # refused, not half-loaded
    assert fresh.get(ds.dataset_id) is None


def test_cancellation_actually_cancels_at_a_stage_boundary(tmp_path: Path) -> None:
    class Gate(StubProvider):
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.proceed = threading.Event()

        def complete(self, request: LLMRequest) -> dict[str, Any]:
            props = (request.schema or {}).get("properties", {})
            if "claims" in props:
                self.entered.set()
                self.proceed.wait(5)  # hold at the 'inferring' stage until released
            return super().complete(request)

    async def scenario() -> None:
        gate = Gate()
        mgr, store = _manager(tmp_path, provider=gate)
        src = _write(tmp_path)
        view = mgr.submit_ingest(source=src, name="orders", idempotency_key="k")
        while not gate.entered.is_set():  # wait until the job is inside the LLM call
            await asyncio.sleep(0.01)
        assert mgr.cancel(view.id) is True
        gate.proceed.set()  # let the call return → next stage boundary sees the cancel
        final = await mgr.wait(view.id)
        assert final.state is JobState.CANCELLED
        assert store.get(dataset_id_for(src, "orders")) is None

    asyncio.run(scenario())


def test_different_datasets_ingest_concurrently(tmp_path: Path) -> None:
    class Probe(StubProvider):
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.release = threading.Event()

        def complete(self, request: LLMRequest) -> dict[str, Any]:
            props = (request.schema or {}).get("properties", {})
            if "claims" in props:
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                self.release.wait(5)
                with self.lock:
                    self.active -= 1
            return super().complete(request)

    async def scenario() -> None:
        probe = Probe()
        mgr, store = _manager(tmp_path, provider=probe)
        a = _write(tmp_path, name="a.csv")
        b = _write(tmp_path, name="b.csv")
        va = mgr.submit_ingest(source=a, name="a", idempotency_key="a")
        vb = mgr.submit_ingest(source=b, name="b", idempotency_key="b")
        while probe.max_active < 2:  # both are inside the LLM call at the same time
            await asyncio.sleep(0.01)
        probe.release.set()
        await asyncio.gather(mgr.wait(va.id), mgr.wait(vb.id))
        assert probe.max_active >= 2  # genuinely parallel, not serialised
        assert store.get(dataset_id_for(a, "a")) is not None
        assert store.get(dataset_id_for(b, "b")) is not None

    asyncio.run(scenario())


def test_query_against_dataset_being_reingested_sees_the_previous_model(tmp_path: Path) -> None:
    class Gate(StubProvider):
        def __init__(self) -> None:
            self.arm = False
            self.entered = threading.Event()
            self.proceed = threading.Event()

        def complete(self, request: LLMRequest) -> dict[str, Any]:
            props = (request.schema or {}).get("properties", {})
            if self.arm and "claims" in props:
                self.entered.set()
                self.proceed.wait(5)
            return super().complete(request)

    async def scenario() -> None:
        gate = Gate()
        mgr, store = _manager(tmp_path, provider=gate)
        src = _write(tmp_path)
        dataset_id = dataset_id_for(src, "orders")
        first = mgr.submit_ingest(source=src, name="orders", idempotency_key="v1")
        await mgr.wait(first.id)
        assert store.get(dataset_id) is not None  # a good model is published

        # a re-ingest (new content key) blocks mid-build; the OLD model stays queryable.
        gate.arm = True
        second = mgr.submit_ingest(source=src, name="orders", idempotency_key="v2")
        while not gate.entered.is_set():
            await asyncio.sleep(0.01)
        assert store.get(dataset_id) is not None  # never a half-built dataset
        gate.proceed.set()
        await mgr.wait(second.id)

    asyncio.run(scenario())


def test_unseen_csv_ingests_in_replay_mode_with_no_credentials(tmp_path: Path) -> None:
    # The zero-credential path (replay, no fixtures) must still ingest an unseen file:
    # role inference falls back to the heuristic rather than dying on a cache miss.
    async def scenario() -> None:
        replay = ReplayProvider(tmp_path / "no-fixtures")
        mgr, store = _manager(tmp_path, provider=replay)
        src = _write(tmp_path)
        view = mgr.submit_ingest(source=src, name="orders", idempotency_key="k")
        final = await mgr.wait(view.id)
        assert final.state is JobState.SUCCEEDED, final.error
        dataset = store.get(dataset_id_for(src, "orders"))
        assert dataset is not None
        assert {b.provenance for b in dataset.model.bindings} == {"heuristic"}
        assert any(m.name == "net_revenue" and m.available for m in dataset.model.measures)

        # a query on an UNSEEN question, however, genuinely needs the LLM — and says so
        # clearly (not a cache-key dump).
        qv = mgr.submit_query(dataset=dataset, question="anything", previous_ir=None)
        q = await mgr.wait(qv.id)
        assert q.state is JobState.FAILED
        assert q.error is not None and q.error.code == "needs_llm"

    asyncio.run(scenario())


def test_query_job_produces_an_answer(tmp_path: Path) -> None:
    async def scenario() -> None:
        mgr, store = _manager(tmp_path)
        src = _write(tmp_path)
        ing = mgr.submit_ingest(source=src, name="orders", idempotency_key="k")
        await mgr.wait(ing.id)
        dataset = store.get(dataset_id_for(src, "orders"))
        assert dataset is not None
        qv = mgr.submit_query(dataset=dataset, question="top products by revenue", previous_ir=None)
        final = await mgr.wait(qv.id)
        assert final.state is JobState.SUCCEEDED
        result = mgr.query_result(qv.id)
        assert result is not None and result.answer is not None
        assert result.answer.rows  # a real answer came back
        assert [e.stage for e in final.events] == ["interpreting", "binding", "executing"]

    asyncio.run(scenario())
