"""Provider seam: replay is hermetic and fails loudly; record round-trips."""

from __future__ import annotations

from pathlib import Path

import pytest

from nl_insights.config import Settings
from nl_insights.provider import (
    LLMRequest,
    RecordingProvider,
    ReplayCacheMiss,
    ReplayProvider,
    build_provider,
)

# A committed smoke fixture proves the replay path end-to-end (see fixtures/llm/).
# Real golden fixtures for the analytical prompts land with the interpreter.
SMOKE_REQUEST = LLMRequest(
    model="claude-sonnet-4-5",
    system="smoke test provider seam",
    prompt="reply with ok=true",
    schema={
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    },
)
SMOKE_RESPONSE = {"ok": True}

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"


class _Canned:
    """A stand-in live provider that returns a fixed response (for tests/recording)."""

    name = "canned"

    def __init__(self, response: dict[str, object]) -> None:
        self._response = response

    def complete(self, request: LLMRequest) -> dict[str, object]:
        return dict(self._response)


def test_cache_key_is_stable_and_order_independent() -> None:
    a = LLMRequest(model="m", system="s", prompt="p", schema={"a": 1, "b": 2})
    b = LLMRequest(model="m", system="s", prompt="p", schema={"b": 2, "a": 1})
    assert a.cache_key() == b.cache_key()  # dict order must not change the key
    assert len(a.cache_key()) == 64  # sha256 hex


def test_committed_smoke_fixture_replays() -> None:
    # The default, credential-free path: a committed fixture is served verbatim.
    provider = ReplayProvider(FIXTURES)
    assert provider.complete(SMOKE_REQUEST) == SMOKE_RESPONSE


def test_replay_fails_loudly_on_miss(tmp_path: Path) -> None:
    provider = ReplayProvider(tmp_path)  # empty dir → every request is a miss
    with pytest.raises(ReplayCacheMiss) as exc:
        provider.complete(SMOKE_REQUEST)
    # The error must be actionable: it names the key and how to record it.
    assert SMOKE_REQUEST.cache_key() in str(exc.value)
    assert "--record" in str(exc.value)


def test_record_then_replay_round_trips(tmp_path: Path) -> None:
    recorder = RecordingProvider(_Canned({"answer": 42}), tmp_path)
    recorded = recorder.complete(SMOKE_REQUEST)
    assert recorded == {"answer": 42}
    # A later replay from the same dir returns the recorded response with no live call.
    assert ReplayProvider(tmp_path).complete(SMOKE_REQUEST) == {"answer": 42}


def test_factory_defaults_to_replay() -> None:
    provider = build_provider(Settings())
    assert provider.name == "replay"


def test_factory_rejects_record_with_replay() -> None:
    with pytest.raises(ValueError, match="record"):
        build_provider(Settings(record=True))


def test_ambient_timeout_is_an_actionable_non_empty_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A live timeout once raised a BARE TimeoutError whose str() is '' — a blank client
    # message. It must instead name the knob and the likely cause. (Guarded: constructing
    # the ambient provider imports the subscription SDK, absent on the replay/CI path.)
    pytest.importorskip("claude_agent_sdk")
    import asyncio

    from nl_insights.provider.ambient import AmbientProvider

    prov = AmbientProvider(timeout_s=0.05)

    async def _slower_than_the_timeout(_request: object) -> dict[str, object]:
        await asyncio.sleep(5)
        return {}

    monkeypatch.setattr(prov, "_complete_async", _slower_than_the_timeout)
    with pytest.raises(TimeoutError) as excinfo:
        prov.complete(SMOKE_REQUEST)
    msg = str(excinfo.value)
    assert msg.strip()  # never blank
    assert "did not respond within" in msg and "NL_INSIGHTS_LLM_TIMEOUT_S" in msg
