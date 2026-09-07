"""The structural guard: no server filesystem path leaves the API boundary, for any error
code, present or future.

The per-case scrubbing (naming the user's own file) protects the errors someone audited. This
pins the invariant instead of the instances: shaped errors are serialised through one redact
function, so a path never crosses the boundary even in an error nobody has written yet. The
rule matches paths ROOTED at a known server directory (the temp dir, the home dir, and the
configured data dir derived lazily from settings), so both failure modes of a shape rule are
gone: a shallow real path (/data/sales.duckdb) is caught because it is under a root, and an API
route (/jobs/abc/status) is not, because /jobs is not a filesystem root we own.

Every test holds two things in tension, because a guard is only real if both hold:
 - it MUST redact a genuine server path - the negative control, plus a data-dir path with NO
   registration call (a guard whose data-dir coverage depended on a startup side effect would
   fail open here);
 - it MUST leave a legitimate message byte-identical, punctuation included - a date, a ratio, a
   slashed column, an API route are path-shaped but are not server paths, and a bracket or full
   stop around a path is not part of it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nl_insights.api import create_app
from nl_insights.config import Settings
from nl_insights.jobs.events import JobError
from nl_insights.provider import LLMRequest
from nl_insights.redact import redact_server_paths

_TMP = tempfile.gettempdir()
_HOME = str(Path.home())

# Fragments a leaked server path would contain; no client-facing message may carry one.
_LEAK_MARKERS = (".uploads", ".staging", "4de0b789e372fd38")


def _leaks(text: str, *extra_roots: str) -> bool:
    roots = (_TMP, _HOME, *extra_roots)
    return any(f"{r}/" in text for r in roots) or any(m in text for m in _LEAK_MARKERS)


# -- negative control: a real path MUST be redacted (fails if redact is a no-op) ---------------


def test_negative_control_a_real_server_path_is_actually_redacted() -> None:
    server = f"{_TMP}/nl-store/.uploads/4de0b789e372fd38.csv"
    before = f'could not load {server} as a CSV: CSV Error in file "{server}" near line 2'
    after = redact_server_paths(before)
    assert after != before  # something changed - this test fails if redact never matches
    assert not _leaks(after)  # no rooted path and no hash survives
    assert "could not load" in after and "near line 2" in after  # the rest is intact


# -- DEFECT A: the data dir is covered WITHOUT any startup registration (no silent fail-open) ---


def test_data_dir_path_redacted_without_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    # A deployment data dir OUTSIDE tmp/home, established only through settings (the env), and no
    # registration call anywhere. If coverage of this root depended on a startup side effect, the
    # guard would silently fail open here and return the path.
    data_dir = "/var/lib/nlinsights/store"
    monkeypatch.setenv("NL_INSIGHTS_DATA_DIR", data_dir)
    assert redact_server_paths(f"built {data_dir}/ds_ab12.duckdb from the upload") == (
        "built [path] from the upload"
    )
    # a two-segment path under it - the exact shape the old depth rule let through
    assert redact_server_paths(f"{data_dir}/x.duckdb") == "[path]"


def test_a_root_in_the_middle_of_a_word_is_not_matched(monkeypatch: pytest.MonkeyPatch) -> None:
    # DEFECT C: the root was spliced in unanchored, so it matched as a SUBSTRING. A relative data
    # dir mangled "metadata" into "meta[path]"; even an absolute one ate "a/data/b". The root is
    # resolved to absolute and matched only at a token boundary now.
    monkeypatch.setenv("NL_INSIGHTS_DATA_DIR", "data")  # a RELATIVE data dir
    assert redact_server_paths("see metadata/schema.json for the shape") == (
        "see metadata/schema.json for the shape"
    )
    monkeypatch.setenv("NL_INSIGHTS_DATA_DIR", "/data")  # absolute, but mid-token must still miss
    assert redact_server_paths("a/data/b relative ref") == "a/data/b relative ref"  # mid-token
    assert redact_server_paths("fail at /data/x.duckdb") == "fail at [path]"  # boundary hit fires


def test_a_path_after_any_separator_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    # DEFECT D: the boundary was an ENUMERATION of preceding chars (whitespace, quote, bracket),
    # so a path introduced by "=" or ":" - key=/path, at:/path - slid through. The boundary is
    # now the CLOSED mid-token set, so every non-token separator is a boundary, including ones
    # not enumerated.
    monkeypatch.setenv("NL_INSIGHTS_DATA_DIR", "/data/nl-insights")
    assert redact_server_paths("dir=/data/nl-insights/x.duckdb not found") == "dir=[path] not found"
    assert redact_server_paths("at:/tmp/x.csv") == "at:[path]"
    assert redact_server_paths("build>/tmp/y.duckdb") == "build>[path]"  # a separator not listed


def test_two_segment_paths_under_the_static_roots_are_redacted() -> None:
    for root in (_TMP, _HOME):
        assert redact_server_paths(f"opening {root}/sales.duckdb failed") == "opening [path] failed"


def test_namedtemporaryfile_shape_and_deep_paths_are_redacted() -> None:
    for p in (
        f"{_TMP}/tmpab12cd9f",  # exactly what NamedTemporaryFile produces
        f"{_TMP}/tmp8sd7f2.csv",
        f"{_HOME}/gt/store/a/b/c/dataset.duckdb",  # four-plus levels
        "boom at .staging/ds_x.duckdb during promote",  # relative internal-store backstop
    ):
        assert not _leaks(redact_server_paths(p)), p


# -- DEFECT B: punctuation AROUND a path is kept, not swallowed by the match --------------------


def test_trailing_punctuation_around_a_path_is_preserved() -> None:
    cases = {
        f"failed ({_TMP}/x.csv) while parsing": "failed ([path]) while parsing",
        f"see [{_TMP}/x.csv] for detail": "see [[path]] for detail",
        f"could not read {_TMP}/x.csv.": "could not read [path].",
        f"read {_TMP}/a.csv, then {_TMP}/b.csv": "read [path], then [path]",
    }
    for inp, expected in cases.items():
        assert redact_server_paths(inp) == expected, inp


# -- over-scrub control: legitimate path-shaped text is byte-identical --------------------------


def test_legitimate_messages_pass_through_byte_identical() -> None:
    for s in (
        "invoice dated 12/1/2010 08:26 could not be parsed",  # a date
        "share is 0.017 of the total",  # a ratio/decimal
        "column margin/revenue is ambiguous",  # a slashed column name
        "poll GET /jobs/abc123/status for progress",  # an API route, not a path
        "the /datasets/xyz/query endpoint expects a body",  # another route
        "growth over 2021-Q1 to 2021-Q2 needs both quarters present",
    ):
        assert redact_server_paths(s) == s, s


def test_one_message_carrying_all_the_traps_survives_intact() -> None:
    # a single message with a date, a ratio, a slashed column, an API route, and the assigned
    # dataset id ds_<digest> - a sha256 of the USER'S OWN upload bytes, a content-derived public
    # handle, NOT a server internal, and not rooted at a server dir - all pass unchanged.
    msg = (
        "dataset ds_4de0b789e372fd38 via GET /jobs/abc123/status on 12/1/2010: the "
        "margin/revenue ratio was 0.017, below the 3/4 threshold"
    )
    assert redact_server_paths(msg) == msg


# -- the structural claim: the choke fires for ANY code, on both dump modes ---------------------


def test_joberror_redacts_message_and_details_for_a_never_seen_code() -> None:
    err = JobError(
        code="a_future_code_nobody_wrote",
        message=f"failed at {_HOME}/gt/store/.staging/ds.duckdb while building the model",
        details={"where": f"{_TMP}/pytest-9/.data/dataset.duckdb", "dataset_id": "sales_ab12"},
    )
    for dumped in (json.dumps(err.model_dump(mode="json")), err.model_dump_json()):
        assert not _leaks(dumped), dumped
        assert "a_future_code_nobody_wrote" in dumped  # the code is preserved
        assert "sales_ab12" in dumped  # a non-path detail value is untouched


# -- the boundary: an induced error through the real API carries no path in the raw JSON --------

_RAGGED = ("\n".join(["a,b,c"] + ["1,2,3"] * 20 + ["9,9,9,9"] + ["1,2,3"] * 20) + "\n").encode()


class _Stub:
    name = "stub"

    def complete(self, request: LLMRequest) -> dict[str, Any]:  # pragma: no cover - not reached
        return {"claims": []}


def test_induced_ingest_error_has_no_server_path_in_the_raw_job_json(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, min_free_bytes=0)
    with TestClient(create_app(settings, provider=_Stub())) as client:
        job = client.post("/datasets", params={"name": "wonky"}, content=_RAGGED).json()
        client.get(f"/jobs/{job['job_id']}/stream")  # drain to terminal
        res = client.get(f"/jobs/{job['job_id']}")
        raw = res.text
        body = res.json()
    assert body["state"] == "failed"
    assert body["error"]["code"] == "ingest_failed"
    assert not _leaks(raw, str(tmp_path)), raw  # tmp_path is under /tmp, so a static root covers it
    assert str(tmp_path) not in raw
    assert "ragged" in body["error"]["message"] or "delimiter" in body["error"]["message"]
