"""Ingestion + profiler on small files with known statistics."""

from __future__ import annotations

from pathlib import Path

import pytest

from nl_insights.ingestion import DateOrder, DialectError, IngestionError, ingest

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_RAW_UCI = _ASSETS / "raw-uci-online-retail.csv"
_ENRICHED = _ASSETS / "nl-insights" / "retail-enriched.csv"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_profile_matches_known_statistics(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        "sales.csv",
        "id,category,region,amount\n"
        "1,A,North,10\n"
        "2,A,North,-5\n"
        "3,B,South,0\n"
        "4,B,South,20\n"
        "5,B,South,\n",  # null amount
    )
    result = ingest(src, data_dir=tmp_path / ".data")
    p = result.profile
    assert p.row_count == 5
    assert p.column_count == 4

    cols = {c.name: c for c in p.columns}

    amount = cols["amount"]
    assert amount.semantic_type == "integer"
    assert amount.null_count == 1
    assert amount.null_rate == pytest.approx(0.2)
    assert amount.distinct_count == 4
    assert amount.min_value == "-5"
    assert amount.max_value == "20"
    assert amount.numeric is not None
    assert (amount.numeric.negatives, amount.numeric.zeros, amount.numeric.positives) == (1, 1, 2)
    assert amount.numeric.integer_ratio == pytest.approx(1.0)

    category = cols["category"]
    assert category.null_count == 0
    assert category.distinct_count == 2
    assert category.top_values[0].value == "B"  # B appears 3×, A twice
    assert category.top_values[0].count == 3

    # id is unique → excluded from FD determinants (its groups are all singletons).
    assert cols["id"].uniqueness_ratio == pytest.approx(1.0)


def test_headerless_file_is_detected_and_profiled(tmp_path: Path) -> None:
    src = _write(tmp_path, "headerless.csv", "1,10,100\n2,20,200\n3,30,300\n")
    result = ingest(src, data_dir=tmp_path / ".data")
    assert result.dialect.has_header is False
    assert result.profile.row_count == 3
    assert result.profile.column_count == 3
    # DuckDB names headerless columns generically; the profile still computes.
    assert result.profile.columns[0].name == "column0"


def test_ambiguous_date_column_resolved_by_evidence(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        "events.csv",
        "event,d\nsignup,01/02/2011\nlogin,13/02/2011\npurchase,05/06/2011\n",
    )
    result = ingest(src, data_dir=tmp_path / ".data")
    assert "d" in result.date_orders
    assert result.date_orders["d"].order is DateOrder.DAY_FIRST  # 13 fixes day-first
    assert "13" in result.date_orders["d"].evidence


def test_semicolon_delimiter_is_sniffed(tmp_path: Path) -> None:
    src = _write(tmp_path, "semi.csv", "a;b;c\n1;2;3\n4;5;6\n")
    result = ingest(src, data_dir=tmp_path / ".data")
    assert result.dialect.delimiter == ";"
    assert result.profile.column_count == 3


def test_ragged_file_fails_loudly(tmp_path: Path) -> None:
    # A clear comma delimiter with one wider row in the MIDDLE — genuine raggedness the
    # sniffer must catch. The filename deliberately does NOT contain 'ragged': the message
    # must diagnose the raggedness itself, not merely echo a filename (the old three-line
    # fixture actually tripped the delimiter detector and only matched 'ragged' because the
    # server path 'ragged.csv' leaked into the message — which it no longer does).
    body = "\n".join(["a,b,c"] + ["1,2,3"] * 20 + ["9,9,9,9"] + ["1,2,3"] * 20) + "\n"
    src = _write(tmp_path, "wonky.csv", body)
    with pytest.raises(DialectError, match="ragged"):
        ingest(src, data_dir=tmp_path / ".data")


def test_header_only_file_is_refused(tmp_path: Path) -> None:
    src = _write(tmp_path, "empty.csv", "a,b,c\n")  # header, no data
    with pytest.raises(IngestionError, match="no data rows"):
        ingest(src, data_dir=tmp_path / ".data")


def test_high_cardinality_transaction_key_is_detected_with_strength(tmp_path: Path) -> None:
    # A transaction key is HIGH cardinality (that is what makes it a key) yet repeats
    # (many rows per invoice). The old cardinality-capped, exact-only detector missed
    # this class entirely; this pins the fix.
    lines = ["invoice,d,item"]
    n_invoices = 12_000  # > the old 10_000 cap: keys must not be excluded by cardinality
    for inv in range(n_invoices):
        day = (inv % 28) + 1
        lines.append(f"INV{inv},2021-01-{day:02d},a")
        # 10 invoices span a second date → invoice→d holds on 11_990/12_000 = 0.9992.
        day2 = day if inv >= 10 else day + 1
        lines.append(f"INV{inv},2021-01-{day2:02d},b")
    src = _write(tmp_path, "txns.csv", "\n".join(lines) + "\n")

    result = ingest(src, data_dir=tmp_path / ".data")
    cols = {c.name: c for c in result.profile.columns}
    assert cols["invoice"].distinct_count == n_invoices  # high cardinality, still a determinant

    fds = {(f.determinant, f.dependent): f for f in result.profile.functional_dependencies}
    assert ("invoice", "d") in fds
    assert fds[("invoice", "d")].strength == pytest.approx(
        0.9992, abs=0.001
    )  # approximate, not lost
    assert fds[("invoice", "d")].support == n_invoices  # measured over every multi-row group


def test_near_unique_numeric_yields_no_spurious_dependency(tmp_path: Path) -> None:
    # A near-unique numeric column (mostly singleton groups) must NOT show a dependency
    # on unrelated columns. Counting singleton groups would inflate strength to ~0.94
    # against anything; strength is measured over multi-row groups only, so pure noise
    # produces nothing. (Reproduces the b2b adversarial file: price ~uniqueness 0.94.)
    import random

    random.seed(7)
    rows = ["price_each,buyer_ref,settle_method"]
    for _ in range(4000):
        price = round(random.uniform(2, 300), 2)  # 2dp collisions → uniqueness ~0.94
        rows.append(f"{price},B{random.randint(1, 900)},{random.choice(['wire', 'card', 'ach'])}")
    src = _write(tmp_path, "b2b.csv", "\n".join(rows) + "\n")

    result = ingest(src, data_dir=tmp_path / ".data")
    price = next(c for c in result.profile.columns if c.name == "price_each")
    assert 0.90 < price.uniqueness_ratio < 0.99  # passes the guard; the singleton fix must catch it
    pairs = {(f.determinant, f.dependent) for f in result.profile.functional_dependencies}
    assert ("price_each", "buyer_ref") not in pairs
    assert ("price_each", "settle_method") not in pairs


@pytest.mark.skipif(not _RAW_UCI.exists(), reason="run `make data` to fetch raw UCI")
def test_raw_uci_surfaces_the_transaction_key(tmp_path: Path) -> None:
    # On the real file, InvoiceNo → InvoiceDate is the strongest transaction-key
    # signal; it must surface as an approximate FD near 0.998, over many groups.
    result = ingest(_RAW_UCI, data_dir=tmp_path / ".data", name="raw_uci")
    fds = {(f.determinant, f.dependent): f for f in result.profile.functional_dependencies}
    fd = fds[("InvoiceNo", "InvoiceDate")]
    assert fd.strength == pytest.approx(0.998, abs=0.005)
    assert fd.support > 10_000  # a real key, measured over thousands of multi-row invoices


@pytest.mark.skipif(not _ENRICHED.exists(), reason="optional local fixture, not in the repo")
def test_enriched_surfaces_a_true_categorical_dependency(tmp_path: Path) -> None:
    result = ingest(_ENRICHED, data_dir=tmp_path / ".data", name="enriched")
    fds = {(f.determinant, f.dependent): f for f in result.profile.functional_dependencies}
    assert fds[("line_type", "is_product_line")].strength == pytest.approx(1.0)


def test_exact_categorical_dependency_with_support(tmp_path: Path) -> None:
    # 6 stores, 3 rows each; each store is in exactly one region → store ⇒ region is
    # exact over 6 multi-row groups (support 6 ≥ the minimum).
    lines = ["store,region,sale"]
    regions = ["East", "East", "East", "West", "West", "West"]
    for s in range(6):
        for k in range(3):
            lines.append(f"S{s},{regions[s]},{10 * k}")
    src = _write(tmp_path, "stores.csv", "\n".join(lines) + "\n")
    result = ingest(src, data_dir=tmp_path / ".data")
    fds = {(f.determinant, f.dependent): f for f in result.profile.functional_dependencies}
    assert fds[("store", "region")].strength == pytest.approx(1.0)
    assert fds[("store", "region")].support == 6
