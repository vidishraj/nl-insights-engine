"""Record/replay providers — the hermetic, credential-free path.

``ReplayProvider`` serves committed fixtures and **fails loudly on a miss**. It never
falls back to a live call: a silent fallback would make CI and the demo quietly
non-hermetic (a passing run that actually hit the network), which is the exact
failure this design exists to prevent. A cache miss is a hard error that tells you
the key and how to record it.

``RecordingProvider`` wraps a live provider and writes each response to the fixture
directory, so ``--record`` produces exactly what ``--replay`` later serves.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from .base import JSONObject, LLMProvider, LLMRequest


class ReplayCacheMiss(RuntimeError):
    """Raised when a replayed request has no committed fixture."""


def _fixture_path(fixtures_dir: Path, request: LLMRequest) -> Path:
    return fixtures_dir / f"{request.cache_key()}.json"


class ReplayProvider:
    """Serve recorded LLM responses from disk; never call out."""

    name = "replay"

    def __init__(self, fixtures_dir: Path) -> None:
        self._dir = fixtures_dir

    def complete(self, request: LLMRequest) -> JSONObject:
        path = _fixture_path(self._dir, request)
        if not path.exists():
            preview = request.prompt.strip().replace("\n", " ")[:120]
            raise ReplayCacheMiss(
                f"No replay fixture for key {request.cache_key()} "
                f"(model={request.model!r}, prompt~={preview!r}). "
                f"Expected file: {path}. "
                f"Record it with the --record flag against a live provider, then commit it. "
                f"Replay never falls back to a live call — that would make this run non-hermetic."
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        response: JSONObject = record["response"]
        return response


class RecordingProvider:
    """Wrap a live provider and persist every response as a replay fixture."""

    name = "record"

    def __init__(self, inner: LLMProvider, fixtures_dir: Path) -> None:
        self._inner = inner
        self._dir = fixtures_dir

    def complete(self, request: LLMRequest) -> JSONObject:
        response = self._inner.complete(request)
        self._dir.mkdir(parents=True, exist_ok=True)
        path = _fixture_path(self._dir, request)
        # Store human-readable request metadata alongside the response so a reviewer
        # can see what each committed fixture answers without decoding the hash.
        record = {
            "request": {
                "model": request.model,
                "system": request.system,
                "prompt": request.prompt,
                "schema": request.schema,
            },
            "response": response,
            # Honest provenance: a replayed cassette should never be mistaken for a
            # hand-written one. 'recorded' + the live provider it came from + model + date.
            "recorded_by": f"recorded ({self._inner.name})",
            "recorded_model": request.model,
            "recorded_at": datetime.date.today().isoformat(),
        }
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return response
