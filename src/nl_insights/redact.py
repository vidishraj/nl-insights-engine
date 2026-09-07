"""Redact server filesystem paths from any message about to cross the API boundary.

A shaped error must carry a stable code and an actionable message but never disclose where
the server keeps its data. Individual call sites already name the user's own file rather than
the server path, but that is a per-case discipline: it protects the errors someone audited
and says nothing about the ones written later. This is the structural backstop - one function,
applied at every point a client-facing message is serialised, so the invariant "no server
path leaves the boundary" holds for error codes that do not exist yet.

It matches what our paths ACTUALLY ARE, not what a path looks like. We know the roots the
server keeps data under - the configured data dir, the temp dir, the home dir - so a path
ROOTED at one of those is redacted at any depth, and nothing else is touched. Recognising a
path by shape instead was both leaky and greedy: a shallow real path (/data/sales.duckdb) slid
through a depth rule while an API route (/jobs/abc/status) got eaten by it.

The data dir is derived LAZILY from settings, not remembered from a startup call. A guard whose
coverage of its most important root depends on a side effect having run elsewhere fails OPEN
when that side effect does not happen - it silently stops guarding, which is the exact class of
"works because of an arrangement elsewhere" bug this change exists to remove. Deriving it here
means there is nothing to forget: any process that serialises an error is covered, worker or CLI
or test helper, without having had to register anything first.
"""

from __future__ import annotations

import re
import tempfile
from functools import lru_cache
from pathlib import Path

# System roots that always hold server files regardless of deployment. The configured data dir
# is added lazily on top (see _current_roots) from settings.
_STATIC_ROOTS: frozenset[str] = frozenset({tempfile.gettempdir(), str(Path.home())})

# A path must begin at a TOKEN BOUNDARY, so a root is matched as the start of a path and not as
# a substring in the middle of a word ("a/data/b" -> "a[path]"). The boundary is defined by the
# CLOSED set - "preceded by a mid-token character": an alphanumeric, '/', '.', '-', '_'. Anything
# else, and the start of the string, is a boundary. This is deliberately the inverse of listing
# the characters that MAY precede a path (whitespace, quote, '=', ':', ',', '>', ...), which is
# an open-ended set that silently leaks a path introduced by a separator nobody enumerated.
_BOUNDARY = r"(?<![A-Za-z0-9/._-])"

# The relative backstop: an internal-store fragment (.uploads/.staging/.data) with no absolute
# root in front of it, e.g. a bare "boom at .staging/ds.duckdb". These dir names are our own
# dotfiles, specific enough that ordinary prose never trips them.
_INTERNAL_DIR = re.compile(_BOUNDARY + r"/?\.(?:uploads|staging|data)/[^\s\"']*")

_PLACEHOLDER = "[path]"

# Punctuation that sits AROUND a path in prose rather than inside it - a closing bracket, a
# sentence stop, a list comma. A greedy match runs to the next space and would swallow these,
# leaving an unbalanced bracket or a dropped full stop in the very message a caller reads when
# something has gone wrong; they are trimmed off the match and kept.
_TRAILING_PUNCT = ".,;:!?)]}"


def _data_root() -> str | None:
    """The deployment's data dir, read from settings at call time (the NL_INSIGHTS_DATA_DIR env,
    else the default). Lazily, so coverage of this root never depends on a startup registration.
    RESOLVED to an absolute path: a relative root is meaningless as a filesystem anchor and, left
    relative, would splice a bare word like 'data' into the pattern and match inside 'metadata'.
    A settings build failure (never expected in a live server) falls back to the static roots."""
    try:
        from .config import Settings

        return str(Path(Settings().data_dir).resolve())
    except Exception:  # pragma: no cover - settings always builds in a live server
        return None


def _current_roots() -> frozenset[str]:
    roots = set(_STATIC_ROOTS)
    data_root = _data_root()
    if data_root:
        roots.add(data_root)
    return frozenset(roots)


@lru_cache(maxsize=16)
def _pattern_for(roots: frozenset[str]) -> re.Pattern[str]:
    # A path rooted at a known dir: a token boundary, the root, then '/', then the rest of the
    # token. Roots are matched longest-first so a nested root wins over a prefix of it; the
    # required '/' after the root means a sibling like /tmpfoo (not under /tmp) is left alone; the
    # boundary means a root in the middle of a word ("a/data/b") is left alone. Keyed by the root
    # set so a change in the configured data dir rebuilds rather than serving a stale pattern.
    alt = "|".join(re.escape(r) for r in sorted(roots, key=len, reverse=True))
    return re.compile(_BOUNDARY + rf"(?:{alt})/[^\s\"']*")


def _replace(match: re.Match[str]) -> str:
    matched = match.group(0)
    stripped = matched.rstrip(_TRAILING_PUNCT)
    return _PLACEHOLDER + matched[len(stripped) :]  # keep trailing punctuation, drop the path


def redact_server_paths(text: str) -> str:
    """Return ``text`` with any server filesystem path replaced by a neutral placeholder.

    Idempotent and safe on any string. Applied to every client-facing error message at the
    serialisation boundary; leaves everything that is not rooted at a known server directory
    (or a relative internal-store fragment) byte-for-byte unchanged, punctuation included.
    """
    if not text:
        return text
    text = _pattern_for(_current_roots()).sub(_replace, text)
    text = _INTERNAL_DIR.sub(_replace, text)
    return text
