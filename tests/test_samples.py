"""The bundled samples: an opaque-id allowlist, the same ingest pipeline, and a safe download.

The security property is the point: an id is a key into a literal dict or it is a 404; no
request string is ever joined to a path. These pin that a traversal id is refused without a
filesystem touch, that a sample runs the SAME pipeline as an upload, that download serves the
same bytes ingest reads, and that a repeat click reuses the dataset instead of duplicating it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from nl_insights.api import create_app
from nl_insights.config import Settings
from nl_insights.provider import LLMRequest, ReplayCacheMiss

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "samples"


class _Heuristic:
    # Forces the credential-free path: role proposal misses the replay cache and the build
    # falls back to the deterministic heuristic, exactly as on a clean clone with no key.
    name = "replay"

    def complete(self, request: LLMRequest) -> dict[str, Any]:
        raise ReplayCacheMiss("no fixture")


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(data_dir=tmp_path, min_free_bytes=0)
    return TestClient(create_app(settings, provider=_Heuristic()))


def test_samples_are_listed_with_label_and_download(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        body = c.get("/samples").json()
    ids = {s["id"] for s in body["samples"]}
    assert {"catering", "storefront", "sensors"} <= ids
    for s in body["samples"]:
        assert s["label"] and s["description"]
        assert s["download"] == f"/samples/{s['id']}/download"


def test_a_traversal_id_is_refused_without_touching_the_filesystem(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        # An encoded traversal never even routes to the endpoint (the id is not one segment);
        # it is a 404 and no filesystem call is reached. This is the whole security property.
        for path in (
            "/samples/..%2F..%2F..%2Fetc%2Fpasswd",
            "/samples/..%2F..%2F..%2Fetc%2Fpasswd/download",
        ):
            r = c.post(path) if not path.endswith("download") else c.get(path)
            assert r.status_code == 404, path
        # A well-formed but unknown id DOES reach the endpoint, and is a shaped 404 from the
        # allowlist miss, never a lookup: an id maps to a spec here or it does not exist.
        assert c.post("/samples/nope").json()["error"]["code"] == "sample_not_found"
        assert c.get("/samples/nope/download").json()["error"]["code"] == "sample_not_found"


def test_loading_a_sample_runs_the_same_pipeline_and_builds_a_model(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        job = c.post("/samples/catering").json()
        assert job["state"] in {"queued", "running", "succeeded"}
        c.get(f"/jobs/{job['job_id']}/stream")  # drain the SAME stage stream to terminal
        final = c.get(f"/jobs/{job['job_id']}").json()
        assert final["state"] == "succeeded", final.get("error")
        # the full pipeline ran: the semantic model exists with real bindings
        model = c.get(f"/datasets/{job['dataset_id']}").json()
    assert model["row_count"] == 202
    roles = {b["column"]: b["role"] for b in model["model"]["bindings"]}
    assert roles.get("order_total") == "monetary_amount"  # the verified stored amount
    assert roles.get("event_date") == "event_time"


def test_download_serves_the_same_bytes_ingest_reads_as_an_attachment(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        r = c.get("/samples/catering/download")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    assert "catering-orders-2024.csv" in r.headers.get("content-disposition", "")
    assert r.content == (_ASSETS / "catering-orders-2024.csv").read_bytes()


def test_a_repeat_click_reuses_the_dataset_rather_than_duplicating(tmp_path: Path) -> None:
    with _client(tmp_path) as c:
        first = c.post("/samples/storefront").json()
        c.get(f"/jobs/{first['job_id']}/stream")
        second = c.post("/samples/storefront").json()  # the second click, seconds later
        listed = c.get("/datasets").json()["datasets"]
    assert first["dataset_id"] == second["dataset_id"]  # same dataset, not a duplicate
    assert first["job_id"] == second["job_id"]  # idempotent: the same job is reused
    assert sum(1 for d in listed if d["dataset_id"] == first["dataset_id"]) == 1
