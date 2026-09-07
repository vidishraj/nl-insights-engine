"""The LLM provider seam.

Every LLM call in the system is a *structured-output* request: the semantic and
interpreter layers never ask the model for free text, they ask it to fill a JSON
schema. That is what makes the output bindable downstream — and it also makes each
call a pure function of ``(model, system, prompt, schema)``, which is exactly the
key we record and replay against.

Three implementations sit behind :class:`LLMProvider`:

* ``ambient``  — Claude subscription auth via the agent SDK (no API key; our dev path)
* ``apikey``   — the Anthropic API with ``ANTHROPIC_API_KEY`` (a grader's own key)
* ``replay``   — committed fixtures, **zero credentials** (the default; hermetic CI + demo)

Replay is the path a grader sees first: clone, one command, real answers, no config.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# A structured response is a plain JSON object; the caller validates it against a
# Pydantic model. Keeping the seam's type this loose means the provider never needs
# to know about the domain models it is serving.
JSONObject = dict[str, Any]


@dataclass(frozen=True)
class LLMRequest:
    """One structured-output request.

    ``schema`` is the JSON schema the response must satisfy (for a live provider it
    becomes a forced tool call whose input *is* the answer). The four fields are the
    whole identity of the call, so their hash is the record/replay cache key.
    """

    model: str
    system: str
    prompt: str
    schema: JSONObject

    def cache_key(self) -> str:
        """Deterministic sha256 over the canonicalised request.

        Sorted keys + tight separators mean the same logical request always hashes
        to the same fixture, regardless of dict ordering or whitespace.
        """
        payload = json.dumps(
            {
                "model": self.model,
                "system": self.system,
                "prompt": self.prompt,
                "schema": self.schema,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn a structured request into a structured response."""

    name: str

    def complete(self, request: LLMRequest) -> JSONObject:
        """Return the structured response for ``request`` (already schema-shaped)."""
        ...
