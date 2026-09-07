"""Generate two of the three bundled sample datasets, reproducibly.

The samples are chosen to demonstrate the taxonomy the design argues, not to be similar:

 - ``storefront-line-items.csv`` has NO total column, so revenue must be DERIVED from
   units times a per-unit price. It exercises the measure algebra doing real work.
 - ``sensor-readings.csv`` is NOT transactional at all: observations with no money
   anywhere. Generic aggregation answers count and average honestly; a revenue question
   refuses, because the concept is not in the file.

The third sample (catering orders, a verified stored amount) is committed as authored.

    uv run python scripts/make_samples.py   # writes both files under assets/samples/

Deterministic (seeded), so the committed CSVs never drift from a rerun.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

_OUT = Path(__file__).resolve().parents[1] / "assets" / "samples"


def _write(name: str, header: list[str], rows: list[list[object]]) -> None:
    path = _OUT / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def storefront_line_items() -> None:
    # Derived revenue: units x price_each, no stored total. A month index rising slightly
    # in volume so a growth question has a real answer; ISO dates, unambiguous by design.
    rng = random.Random(4021)
    catalogue = [
        ("Garden", "Terracotta Pot", 6.5),
        ("Garden", "Trowel", 9.0),
        ("Garden", "Watering Can", 14.5),
        ("Kitchen", "Chopping Board", 12.0),
        ("Kitchen", "Enamel Mug", 7.5),
        ("Kitchen", "Cast Pan", 38.0),
        ("Hardware", "Tape Measure", 8.0),
        ("Hardware", "Screwdriver Set", 22.0),
        ("Hardware", "LED Bulb", 4.5),
        ("Stationery", "Notebook", 5.0),
        ("Stationery", "Ink Pen", 3.5),
    ]
    header = ["line_id", "sale_date", "category", "product", "units", "price_each"]
    rows: list[list[object]] = []
    lid = 5000
    for month in range(1, 13):
        volume = 10 + month  # a gentle upward trend across the year
        for _ in range(volume):
            cat, prod, price = rng.choice(catalogue)
            day = rng.randint(1, 28)
            units = rng.randint(1, 12)
            jitter = rng.choice([-0.5, 0.0, 0.0, 0.5])
            rows.append(
                [
                    f"L{lid}",
                    f"2024-{month:02d}-{day:02d}",
                    cat,
                    prod,
                    units,
                    f"{price + jitter:.2f}",
                ]
            )
            lid += 1
    _write("storefront-line-items.csv", header, rows)


def sensor_readings() -> None:
    # Not transactional: measurements with no money. count and average are honest; a
    # revenue question has no concept to bind and must refuse.
    rng = random.Random(7731)
    stations = ["ST-ALPHA", "ST-BRAVO", "ST-CHARLIE", "ST-DELTA"]
    header = [
        "station",
        "reading_date",
        "temperature_c",
        "humidity_pct",
        "wind_kph",
        "pressure_hpa",
    ]
    rows: list[list[object]] = []
    for day in range(1, 61):  # two months of daily readings per station
        for st in stations:
            temp = round(rng.uniform(-4.0, 31.0), 1)
            hum = rng.randint(28, 98)
            wind = round(rng.uniform(2.0, 46.0), 1)
            # a decimal, so temperature/wind/pressure are THREE decimal-bearing columns: the
            # heuristic abstains from a money role rather than guessing one, which is exactly
            # right here - there is no money in the file, so a revenue question has nothing to
            # bind and refuses, while count and average over the measurements answer honestly.
            pres = round(rng.uniform(985.0, 1035.0), 1)
            rows.append([st, f"2024-03-{day:02d}" if day <= 31 else f"2024-04-{day - 31:02d}",
                         temp, hum, wind, pres])
    _write("sensor-readings.csv", header, rows)


if __name__ == "__main__":
    _OUT.mkdir(parents=True, exist_ok=True)
    storefront_line_items()
    sensor_readings()
