"""Fetch the sample datasets. This is the documented, one-command download step.

The raw UCI file is fetched (large, so ``assets/*.csv`` is gitignored); the synthetic slice
is committed and needs no download:

* **Raw UCI Online Retail** — downloaded from the UCI archive and converted from
  ``.xlsx`` to CSV with ``all_varchar`` so the *original* strings survive (the
  ``dd/mm/yyyy`` invoice dates and ``C``-prefixed cancellations that make it a good
  adversarial second fixture).

Run: ``uv run python scripts/fetch_datasets.py``  (or ``make data``).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
RAW_CSV = ASSETS / "raw-uci-online-retail.csv"


def fetch_raw_uci() -> None:
    if RAW_CSV.exists():
        print(f"raw UCI already present: {RAW_CSV}")
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    xlsx_path = ASSETS / "online-retail.xlsx"
    if not xlsx_path.exists():
        print(f"downloading {UCI_URL} ...")
        with urllib.request.urlopen(UCI_URL, timeout=120) as resp:  # noqa: S310 - trusted archive
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            xlsx_name = next(n for n in archive.namelist() if n.lower().endswith(".xlsx"))
            xlsx_path.write_bytes(archive.read(xlsx_name))
    con = duckdb.connect()
    con.execute("INSTALL excel; LOAD excel")
    # all_varchar reads every cell as its raw string — essential because InvoiceNo
    # mixes numbers and 'C'-prefixed cancellations, which would break per-column type
    # inference (the very trap our profiler is built to catch). The one exception is
    # the date column: Excel stores it as a numeric serial, so we render it to the
    # canonical m/d/yyyy string the real-world CSV export shows (which also gives our
    # date-order resolver a genuinely ambiguous column to reason about).
    # COPY's paths must be inlined literals (they are not bind-parameterisable).
    xlsx_lit = str(xlsx_path).replace("'", "''")
    csv_lit = str(RAW_CSV).replace("'", "''")
    to_ts = "to_timestamp((CAST(\"InvoiceDate\" AS DOUBLE) - 25569) * 86400) AT TIME ZONE 'UTC'"
    con.execute(
        f"COPY (SELECT * REPLACE (strftime({to_ts}, '%m/%d/%Y %H:%M') AS \"InvoiceDate\") "
        f"FROM read_xlsx('{xlsx_lit}', header=true, all_varchar=true)) "
        f"TO '{csv_lit}' (HEADER, DELIMITER ',')"
    )
    print(f"wrote {RAW_CSV}")  # the cached .xlsx is left in place (gitignored) for re-runs


if __name__ == "__main__":
    fetch_raw_uci()
