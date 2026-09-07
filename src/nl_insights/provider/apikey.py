"""API-key provider — the Anthropic API via ``ANTHROPIC_API_KEY``.

Structured output is forced with a single tool: the model *must* call ``respond``
whose ``input_schema`` is the request's schema, and the tool input it returns *is*
the structured answer. There is no free-text parsing path.

The ``anthropic`` SDK is imported lazily so the default replay path (and CI) need
neither the package nor a key.
"""

from __future__ import annotations

import os

from .base import JSONObject, LLMRequest

_TOOL_NAME = "respond"


class ApiKeyProvider:
    name = "apikey"

    def __init__(self, api_key: str | None = None, *, max_tokens: int = 4096) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ApiKeyProvider needs ANTHROPIC_API_KEY. Set it, or use --replay for a "
                "credential-free run."
            )
        try:
            from anthropic import Anthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "The apikey provider needs the 'anthropic' package: install the extra "
                "(uv sync --extra apikey) or use --replay."
            ) from exc
        self._client = Anthropic(api_key=key)
        self._max_tokens = max_tokens

    def complete(self, request: LLMRequest) -> JSONObject:
        message = self._client.messages.create(
            model=request.model,
            max_tokens=self._max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": request.prompt}],
            tools=[
                {
                    "name": _TOOL_NAME,
                    "description": "Return the answer as structured data matching the schema.",
                    "input_schema": request.schema,
                }
            ],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                return dict(block.input)
        raise RuntimeError("Model did not return the forced structured tool call.")
