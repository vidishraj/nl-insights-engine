"""Dialect sniffing — determine encoding, delimiter, quote char, and header presence.

Everything here is deterministic and *fails loudly*: if the delimiter can't be
inferred, or the rows are ragged, we raise a specific error rather than let a wrong
guess flow silently into the rest of the pipeline. The sniffed dialect is then handed
to DuckDB explicitly, so DuckDB is never left to re-guess.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

_SAMPLE_BYTES = 128 * 1024


class DialectError(ValueError):
    """Raised when the CSV dialect cannot be determined with confidence."""


def caller_label(display_name: str | None) -> str:
    """The name to show a CLIENT for the file under inspection. A shaped error must never
    disclose the server path or the internal content-hash filename (they leak the data-store
    layout); reference the user's own filename, or a neutral phrase when we were handed none.
    The real server path belongs in the operator log, not the API response."""
    return display_name or "the uploaded file"


def _last_record_boundary(text: str, quotechar: str) -> int:
    """Index of the last newline that ends a COMPLETE record — a newline that falls OUTSIDE a
    quoted field. A multi-line quoted field contains newlines that are NOT record boundaries;
    truncating a 128 KB sample at a raw newline can split such a field, leaving an unbalanced
    quote that makes a perfectly valid file look ragged (the file DuckDB reads fine). Trimming
    to the last real record boundary before the rectangularity check removes that false
    positive. Returns -1 when no complete record ends within the text (a single field larger
    than the sample) — the caller then declines to judge raggedness and lets DuckDB arbitrate."""
    in_quote = False
    boundary = -1
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == quotechar:
            if in_quote and i + 1 < n and text[i + 1] == quotechar:
                i += 2  # a doubled "" inside a quoted field is an escaped quote, not a close
                continue
            in_quote = not in_quote
        elif ch == "\n" and not in_quote:
            boundary = i
        i += 1
    return boundary


@dataclass(frozen=True)
class Dialect:
    encoding: str
    delimiter: str
    quotechar: str
    has_header: bool


def _detect_encoding(raw: bytes) -> str:
    """BOM first, then UTF-8, then cp1252. cp1252 decodes any byte, so it is the
    documented last resort (retail exports like raw UCI are Latin-1/cp1252)."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1252"


def sniff_dialect(path: Path, *, display_name: str | None = None) -> Dialect:
    label = caller_label(display_name)
    raw = path.read_bytes()[:_SAMPLE_BYTES]
    if not raw.strip():
        raise DialectError(f"{label} is empty or whitespace-only.")

    encoding = _detect_encoding(raw)
    text = raw.decode(encoding, errors="replace")

    # Trim a trailing partial line so the sniffer sees only whole rows.
    if "\n" in text:
        text = text[: text.rindex("\n")]

    try:
        dialect = csv.Sniffer().sniff(text, delimiters=",;\t|")
    except csv.Error as exc:
        raise DialectError(
            f"Could not determine the delimiter for {label}: {exc}. "
            f"Supported delimiters are comma, semicolon, tab, and pipe."
        ) from exc

    try:
        has_header = csv.Sniffer().has_header(text)
    except csv.Error:
        # Ambiguous header detection isn't fatal — default to header-present, which is
        # the overwhelming norm for the transactional CSVs this serves, and record it.
        has_header = True

    delimiter = dialect.delimiter
    quotechar = dialect.quotechar or '"'

    # Validate rectangularity on COMPLETE records only. The sample may have been truncated at
    # 128 KB mid quoted-field; judging raggedness over a split field rejects a valid file AND
    # misattributes it to the delimiter. Trim to the last real record boundary first; if the
    # sample is one unterminated record (a field bigger than the sample), decline to judge and
    # let DuckDB's full-file parse arbitrate rather than reject on a fragment.
    boundary = _last_record_boundary(text, quotechar)
    check_text = text[:boundary] if boundary > 0 else None
    if check_text is not None:
        reader = csv.reader(io.StringIO(check_text), delimiter=delimiter, quotechar=quotechar)
        rows = [r for r in reader if r]
        if not rows:
            raise DialectError(f"No parseable rows in {label} with delimiter {delimiter!r}.")
        width = len(rows[0])
        for i, row in enumerate(rows[1:], start=2):
            if len(row) != width:
                # State the fact (ragged), not an unestablished cause. It MAY be a mis-detected
                # delimiter or quoting; do not assert one we have not confirmed.
                raise DialectError(
                    f"{label} is ragged near line {i}: expected {width} fields, got {len(row)} "
                    f"with delimiter {delimiter!r} and quote {quotechar!r} — the delimiter or "
                    "quoting may be mis-detected."
                )

    return Dialect(
        encoding=encoding, delimiter=delimiter, quotechar=quotechar, has_header=has_header
    )
