"""The HTTP surface: boots credential-free, and the full ingest -> stream -> query flow.

A stub provider is injected so the endpoints run with no fixtures or credentials. The
TestClient is used as a CONTEXT MANAGER so one event loop persists across requests and
background jobs make progress; the SSE stream is drained synchronously (TestClient reads
the whole body) to await each job. Errors are asserted to be SHAPED (a code + message),
never a stack trace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from nl_insights.api import create_app
from nl_insights.config import Settings
from nl_insights.provider import LLMRequest

_ORDERS = (
    "order_id,ts,product,qty,price,customer\n"
    + "".join(
        f"O{o},2021-01-{o + 1:02d},P{k},{2 + k},{1.5 + k},C{o}\n"
        for o in range(8)
        for k in range(3)
    )
).encode()

_CLAIMS = {
    "order_id": ("transaction_key", None),
    "ts": ("event_time", None),
    "product": ("entity_key", "product"),
    "qty": ("additive_quantity", None),
    "price": ("monetary_rate", None),
    "customer": ("entity_key", "customer"),
}


class StubProvider:
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
        return {
            "measures": ["net_revenue"],
            "group_by": ["product"],
            "top_k": {"measure": "net_revenue", "k": 3},
        }


def _client(tmp_path: Path) -> TestClient:
    # Disable the production disk floor here: CI free space is not the thing under test, and
    # the guard has its own dedicated test. (The default floor is exercised by
    # test_low_disk_declines_upload_with_a_shaped_507.)
    settings = Settings(data_dir=tmp_path, min_free_bytes=0)
    return TestClient(create_app(settings, provider=StubProvider()))


def _ingest(client: TestClient, name: str = "orders") -> dict[str, Any]:
    """POST the CSV and drain the SSE stream so the dataset is ready on return."""
    job = client.post("/datasets", params={"name": name}, content=_ORDERS).json()
    client.get(f"/jobs/{job['job_id']}/stream")  # blocks until terminal
    return job


def test_health_reports_replay_by_default() -> None:
    client = TestClient(create_app(Settings()))
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["provider"] == "replay"  # a clean machine gets a working endpoint


def test_ingest_stream_and_query_end_to_end(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        res = client.post("/datasets", params={"name": "orders"}, content=_ORDERS)
        assert res.status_code == 202
        job = res.json()
        assert job["state"] == "queued"
        jid, dataset_id = job["job_id"], job["dataset_id"]

        # stream the stage events (draining the SSE body awaits completion)
        stream = client.get(f"/jobs/{jid}/stream")
        assert stream.headers["content-type"].startswith("text/event-stream")
        for stage in ["sniffing", "profiling", "inferring", "verifying", "promoting"]:
            assert stage in stream.text
        assert "event: done" in stream.text

        assert client.get(f"/jobs/{jid}").json()["state"] == "succeeded"

        # the dataset's understanding is inspectable
        model = client.get(f"/datasets/{dataset_id}").json()
        assert model["row_count"] == 24
        assert any(m["name"] == "net_revenue" for m in model["model"]["measures"])

        # query -> 202 -> stream -> answer
        q = client.post(
            f"/datasets/{dataset_id}/query", json={"question": "top products by revenue"}
        )
        assert q.status_code == 202
        qid = q.json()["job_id"]
        qstream = client.get(f"/jobs/{qid}/stream")
        assert "binding" in qstream.text  # the plan passed the binder
        result = client.get(f"/jobs/{qid}").json()["result"]
        assert result["kind"] in {"answerable", "answer_with_caveats"}
        # ONE finalised class, read by every consumer: the class the payload reports is exactly
        # the one its own caveats imply. A bind-time kind out of step with the answer's caveats
        # would show up right here as a mismatch.
        expected = "answer_with_caveats" if result["answer"]["caveats"] else "answerable"
        assert result["kind"] == expected
        assert result["answer"]["rows"]


def test_multipart_upload_ingests_correctly(tmp_path: Path) -> None:
    # curl -F 'file=@orders.csv' and browser pickers send multipart — Starlette parses
    # it, and we must ingest the FILE, not the envelope.
    with _client(tmp_path) as client:
        res = client.post(
            "/datasets",
            params={"name": "orders"},
            files={"file": ("orders.csv", _ORDERS, "text/csv")},
        )
        assert res.status_code == 202
        job = res.json()
        client.get(f"/jobs/{job['job_id']}/stream")  # await
        state = client.get(f"/jobs/{job['job_id']}").json()
        assert state["state"] == "succeeded", state.get("error")
        model = client.get(f"/datasets/{job['dataset_id']}").json()
        assert model["row_count"] == 24  # the CSV rows, not envelope bytes


def test_dataset_survives_a_restart_and_still_answers(tmp_path: Path) -> None:
    # The acceptance that matters: ingest, "restart" the service (a fresh app on the same
    # data_dir → an empty in-memory store until rehydration), and the dataset must still
    # ANSWER a query — not merely appear in /datasets.
    with _client(tmp_path) as client:
        job = client.post("/datasets", params={"name": "orders"}, content=_ORDERS).json()
        client.get(f"/jobs/{job['job_id']}/stream")  # await ready
        dataset_id = job["dataset_id"]
        assert client.get(f"/datasets/{dataset_id}").status_code == 200

    with _client(tmp_path) as restarted:  # a new process would build a new empty store
        listed = [d["dataset_id"] for d in restarted.get("/datasets").json()["datasets"]]
        assert dataset_id in listed  # rehydrated from its sidecar on startup
        q = restarted.post(
            f"/datasets/{dataset_id}/query", json={"question": "top products by revenue"}
        )
        assert q.status_code == 202
        qid = q.json()["job_id"]
        restarted.get(f"/jobs/{qid}/stream")  # await
        result = restarted.get(f"/jobs/{qid}").json()["result"]
        assert result["kind"] in {"answerable", "answer_with_caveats"}
        expected = "answer_with_caveats" if result["answer"]["caveats"] else "answerable"
        assert result["kind"] == expected  # the payload's class matches its own caveats
        assert result["answer"]["rows"]  # a real SQL answer over the persisted DuckDB file


def test_raw_body_that_is_actually_multipart_is_diagnosed(tmp_path: Path) -> None:
    envelope = (
        b"--BOUNDARY\r\n"
        b'Content-Disposition: form-data; name="file"; filename="x.csv"\r\n'
        b"Content-Type: text/csv\r\n\r\n"
        b"a,b\n1,2\n--BOUNDARY--\r\n"
    )
    with _client(tmp_path) as client:
        # sent as a raw body (not multipart content-type) → we diagnose it, not the sniffer
        res = client.post(
            "/datasets", content=envelope, headers={"content-type": "application/octet-stream"}
        )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "looks_like_multipart"


def test_repeated_identical_upload_is_idempotent(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        a = client.post("/datasets", params={"name": "orders"}, content=_ORDERS).json()
        b = client.post("/datasets", params={"name": "orders"}, content=_ORDERS).json()
        assert a["job_id"] == b["job_id"]  # not ingested twice


def test_unknown_job_is_a_shaped_404(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        res = client.get("/jobs/does-not-exist")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "job_not_found"
    assert "Traceback" not in res.text  # no stack trace ever


def test_empty_upload_is_rejected(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        res = client.post("/datasets", content=b"")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "empty_upload"


def test_over_cap_upload_is_a_shaped_413(tmp_path: Path) -> None:
    # the resource guard rejects an over-cap upload rather than buffering it — protecting a
    # public endpoint's shared disk. A tiny cap makes the sample body over-cap.
    app = create_app(Settings(data_dir=tmp_path, max_upload_bytes=8), provider=StubProvider())
    with TestClient(app) as client:
        res = client.post("/datasets", params={"name": "orders"}, content=_ORDERS)
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "upload_too_large"


def test_low_disk_declines_upload_with_a_shaped_507(tmp_path: Path) -> None:
    # a free-space floor larger than any real disk forces the guard to decline — a safe,
    # correct outcome that protects datasets already on the box, not an out-of-disk crash.
    app = create_app(Settings(data_dir=tmp_path, min_free_bytes=10**18), provider=StubProvider())
    with TestClient(app) as client:
        res = client.post("/datasets", params={"name": "orders"}, content=_ORDERS)
    assert res.status_code == 507
    assert res.json()["error"]["code"] == "storage_full"


def test_query_on_unknown_dataset_is_404(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        res = client.post("/datasets/nope/query", json={"question": "revenue"})
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "dataset_not_found"


def test_followup_referencing_a_bad_prior_job_is_409(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        job = _ingest(client)
        res = client.post(
            f"/datasets/{job['dataset_id']}/query",
            json={"question": "and last month?", "previous_job_id": "missing"},
        )
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "previous_job_unavailable"


def test_cancel_unknown_job_is_404(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        res = client.post("/jobs/nope/cancel")
    assert res.status_code == 404


def test_glass_box_ui_is_served_at_root(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    # the page exposes internals, not a chat skin
    assert "glass box" in res.text
    assert "Ingest" in res.text and "verifiers" in res.text


def test_malformed_query_body_uses_the_unified_error_shape(tmp_path: Path) -> None:
    # a validation error must return our {error:{code,message,details}} contract, not
    # FastAPI's default {detail:[...]} — one error shape across the whole API.
    with _client(tmp_path) as client:
        res = client.post("/datasets/any/query", json={"not_question": "x"})  # missing 'question'
    assert res.status_code == 422
    body = res.json()
    assert "error" in body and body["error"]["code"] == "invalid_request"
    assert "detail" not in body
