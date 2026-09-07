"""Build the configured :class:`LLMProvider` from :class:`Settings`.

This is the single place that maps a mode to an implementation, so the rest of the
system depends only on the seam, never on which auth path is live.
"""

from __future__ import annotations

from ..config import Settings
from .base import LLMProvider
from .replay import RecordingProvider, ReplayProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.provider == "replay":
        if settings.record:
            raise ValueError("--record needs a live provider (apikey or ambient), not replay.")
        return ReplayProvider(settings.fixtures_dir)

    # Live paths — imported lazily so replay never needs their SDKs.
    live: LLMProvider
    if settings.provider == "apikey":
        from .apikey import ApiKeyProvider

        live = ApiKeyProvider()
    elif settings.provider == "ambient":
        from .ambient import AmbientProvider

        live = AmbientProvider(timeout_s=settings.llm_timeout_s)
    else:  # pragma: no cover - Literal makes this unreachable
        raise ValueError(f"unknown provider {settings.provider!r}")

    return RecordingProvider(live, settings.fixtures_dir) if settings.record else live
