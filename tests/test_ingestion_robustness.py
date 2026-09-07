"""wb-wkn.46 — front-door ingestion robustness on files DuckDB itself reads fine.

'It would not load my file' is worse than any wrong number, and the walkthrough hands over
an unseen CSV. These pin two loader bugs and keep a genuinely-broken file refusing:
 - a multi-line quoted field that straddles the 128 KB sniffer sample (the sample truncated
   mid-quote, so the rectangularity check saw an unbalanced quote and rejected a valid file);
 - a cp1252/Latin-1 file (the sniffed encoding was never passed to DuckDB, so it died on UTF-8).
Generated inline (the committed regression guard) rather than shipping a 382 KB binary fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nl_insights.ingestion import ingest
from nl_insights.ingestion.dialect import DialectError
from nl_insights.ingestion.loader import IngestionError


def test_multiline_quoted_field_past_the_sniffer_sample_ingests(tmp_path: Path) -> None:
    src = tmp_path / "multiline_big.csv"
    # > 128 KB so the sniffer's 128 KB sample truncates mid-file; every row carries a quoted
    # field with embedded newlines, so the truncation lands inside a quoted field.
    para = '"' + ("lorem ipsum dolor sit amet, line one\nline two of the note\n" * 8) + '"'
    lines = ["id,note,amount,category"]
    for i in range(400):
        lines.append(f"R{i},{para},{i * 1.5},cat{i % 4}")
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert src.stat().st_size > 128 * 1024  # genuinely past the sniffer sample

    result = ingest(src, data_dir=tmp_path / ".data", name="ml")
    assert result.profile.row_count == 400  # the whole valid file, not a rejection
    assert len(result.profile.columns) == 4


def test_cp1252_latin1_file_ingests_via_the_sniffed_encoding(tmp_path: Path) -> None:
    src = tmp_path / "cp1252.csv"
    # 'café' and a cp1252 smart quote (0x92) — bytes that are INVALID UTF-8, so a UTF-8 read
    # dies; the sniffer detects cp1252 and the loader must pass it through to DuckDB.
    rows = ["id,name,price"]
    for i in range(60):
        rows.append(f"P{i},café ’{i}’,{i + 0.5}")
    src.write_bytes(("\n".join(rows) + "\n").encode("cp1252"))

    result = ingest(src, data_dir=tmp_path / ".data", name="cp")
    assert result.dialect.encoding == "cp1252"
    assert result.profile.row_count == 60
    import duckdb

    con = duckdb.connect(str(result.duckdb_path))
    try:
        val = con.execute(f"SELECT name FROM {result.table} LIMIT 1").fetchone()[0]
        assert "caf" in val and "é" in val  # the accented char round-tripped, not mojibake
    finally:
        con.close()


def test_a_genuinely_ragged_file_still_refuses_with_an_accurate_reason(tmp_path: Path) -> None:
    src = tmp_path / "ragged.csv"
    # unambiguous comma delimiter, but one row has an extra field — a real structural fault the
    # loader must still refuse (the fix must not make it accept mangled data).
    lines = ["a,b,c"] + ["1,2,3"] * 40 + ["9,9,9,9"] + ["1,2,3"] * 40
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises((DialectError, IngestionError)) as exc:
        ingest(src, data_dir=tmp_path / ".data", name="rg")
    assert "ragged" in str(exc.value).lower() or "delimiter" in str(exc.value).lower()


def test_ambient_error_detail_is_never_empty() -> None:
    # message.errors is often None (present but null) -> 'ambient run failed: None'. The detail
    # must fall through to something non-empty rather than render a bare None.
    from types import SimpleNamespace

    from nl_insights.provider.ambient import _error_detail

    empty = SimpleNamespace(is_error=True, errors=None, subtype=None)
    detail = _error_detail(empty, None)
    assert detail and detail.strip() and detail != "None"
    # and it prefers real detail when present
    assert _error_detail(SimpleNamespace(errors="rate limited"), None) == "rate limited"
    assert _error_detail(SimpleNamespace(errors=None, subtype="error_max_turns"), None) == (
        "error_max_turns"
    )


# -- wb-wkn.47: a shaped error must never disclose the server path or the internal hash ---------

# The internal directory names a refusal must never expose (they map the data store); a test
# for "no filesystem separator followed by a known internal directory name" checks these.
_INTERNAL_DIRS = (".uploads", ".staging", ".data")


def _leaks_a_server_path(message: str) -> bool:
    import os

    sep = os.sep
    return any(sep + d in message for d in _INTERNAL_DIRS)


def test_dialect_error_names_the_users_file_not_the_server_path(tmp_path: Path) -> None:
    # sniff_dialect is handed the server path but must render the user's filename in its refusal.
    from nl_insights.ingestion.dialect import DialectError, sniff_dialect

    # a genuinely undelimitable sample, stored under a hash name inside .uploads
    uploads = tmp_path / ".uploads"
    uploads.mkdir()
    server_src = uploads / "4de0b789e372fd38.csv"
    server_src.write_text("a single column\nwith no delimiter at all\njust prose lines\n")

    with pytest.raises(DialectError) as exc:
        sniff_dialect(server_src, display_name="quarterly_sales.csv")
    msg = str(exc.value)
    assert "quarterly_sales.csv" in msg  # the user's own filename
    assert "4de0b789e372fd38" not in msg  # never the internal content hash
    assert str(server_src) not in msg and not _leaks_a_server_path(msg)
    # the actionable guidance survives verbatim
    assert "comma, semicolon, tab, and pipe" in msg


def test_scrubbers_strip_the_server_path_and_hash() -> None:
    from nl_insights.ingestion.loader import _scrub_source_path
    from nl_insights.jobs.pipeline import _scrub_internal_paths

    source = Path("/data/nl-insights/.uploads/4de0b789e372fd38.csv")
    duck = f'CSV Error on Line: 2 ... in file "{source}"'
    scrubbed = _scrub_source_path(duck, source, "orders.csv")
    assert str(source) not in scrubbed and "4de0b789e372fd38" not in scrubbed
    assert "orders.csv" in scrubbed

    staging = Path("/data/nl-insights/.staging/ds_x.duckdb")
    data = Path("/data/nl-insights/.data")
    net = _scrub_internal_paths(f"boom at {source} and {staging} under {data}", "orders.csv",
                                source, staging, data)
    assert not _leaks_a_server_path(net)
    assert str(source) not in net and str(staging) not in net and str(data) not in net


def test_shaped_ingest_error_carries_no_server_path_or_hash(tmp_path: Path) -> None:
    # The full front-door path: a genuinely ragged file uploaded to the server's .uploads dir
    # must fail with a SHAPED error that names the user's file and keeps the delimiter guidance,
    # but exposes neither the server path, the .uploads/.staging/.data layout, nor the hash.
    from types import SimpleNamespace

    from nl_insights.jobs.pipeline import PipelineError, run_ingest

    uploads = tmp_path / ".uploads"
    uploads.mkdir()
    server_src = uploads / "4de0b789e372fd38.csv"
    # clear comma delimiter, but one row is wider — ragged, a real structural fault.
    rows = ["a,b,c"] + ["1,2,3"] * 30 + ["9,9,9,9"] + ["1,2,3"] * 30
    server_src.write_text("\n".join(rows) + "\n")

    provider = SimpleNamespace(name="stub")  # never reached: sniff refuses before any LLM call
    with pytest.raises(PipelineError) as exc:
        run_ingest(
            source=server_src,
            name="ds",
            provider=provider,  # type: ignore[arg-type]
            model="m",
            data_dir=tmp_path / ".data",
            staging_dir=tmp_path / ".staging",
            emit=lambda _s, _m: None,
            display_name="my_quarterly_report.csv",
        )
    message = exc.value.error.message
    assert exc.value.error.code == "ingest_failed"
    assert "my_quarterly_report.csv" in message  # the user's filename
    assert "4de0b789e372fd38" not in message  # no internal content hash
    assert str(server_src) not in message  # no absolute server path
    assert not _leaks_a_server_path(message)  # no /.uploads, /.staging, /.data
    assert "ragged" in message or "delimiter" in message  # the true, actionable reason survives
