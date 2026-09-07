"""Ambient provider — Claude subscription auth via the agent SDK, no API key.

This is the development path: it reuses the local Claude credentials (the ``claude`` CLI
the SDK drives) rather than a billed API key. It is the *least* depended-upon path — CI
and graders use replay — so the SDK is imported lazily and a missing SDK / CLI is a
clear, actionable error rather than an import-time crash.

Structured output: the agent SDK exposes an agentic surface, not the raw messages API's
``tool_choice``. We get schema-shaped JSON by (1) preferring the SDK's own
``structured_output`` when the run populates it, and (2) otherwise instructing the model
to emit ONLY a JSON object conforming to the request's schema and extracting it. Either
way the caller validates the result against its Pydantic model, so a malformed response
fails loudly at the binding site rather than flowing on.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .base import JSONObject, LLMRequest

_JSON_INSTRUCTION = (
    "\n\nReturn ONLY a single JSON object conforming to this JSON Schema — no prose, no "
    "markdown fences, no explanation:\n"
)


class AmbientProvider:
    name = "ambient"

    def __init__(self, *, max_turns: int = 1, timeout_s: float = 600.0) -> None:
        self._timeout_s = timeout_s
        try:
            import claude_agent_sdk  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "The ambient provider needs 'claude-agent-sdk' (subscription auth): install "
                "the extra (uv sync --extra ambient), set ANTHROPIC_API_KEY for the apikey "
                "path, or use --replay for a credential-free run."
            ) from exc
        self._sdk = claude_agent_sdk
        self._max_turns = max_turns

    def complete(self, request: LLMRequest) -> JSONObject:  # pragma: no cover - needs live auth
        # complete() is sync (the whole seam is); the SDK is async. Callers run inside a
        # worker thread (jobs use to_thread) or a plain script, so there is no running
        # loop to clash with — asyncio.run is safe here. A TIMEOUT is essential: without
        # it a hung `claude` CLI leaves a query job stuck in 'interpreting' forever, and
        # cancellation (stage-boundary cooperative) cannot land mid-provider-call. On
        # expiry, translate the BARE TimeoutError (whose str() is '') into an actionable,
        # non-empty message — otherwise the shaped error downstream carries nothing.
        try:
            return asyncio.run(
                asyncio.wait_for(self._complete_async(request), timeout=self._timeout_s)
            )
        except TimeoutError as exc:  # asyncio.TimeoutError is an alias since 3.11
            raise TimeoutError(
                f"the model did not respond within {self._timeout_s:.0f}s — the file may be "
                "unusually wide (many columns build a large prompt); retry, or raise "
                "NL_INSIGHTS_LLM_TIMEOUT_S"
            ) from exc

    async def _complete_async(self, request: LLMRequest) -> JSONObject:  # pragma: no cover
        sdk = self._sdk
        options = sdk.ClaudeAgentOptions(
            system_prompt=request.system,
            model=request.model,
            max_turns=self._max_turns,
            # a pure completion — the model must not touch the filesystem or run tools.
            allowed_tools=[],
            permission_mode="default",
        )
        prompt = request.prompt + _JSON_INSTRUCTION + json.dumps(request.schema)

        text_parts: list[str] = []
        result_text: str | None = None
        structured: Any = None
        async for message in sdk.query(prompt=prompt, options=options):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, sdk.ResultMessage):
                result_text = getattr(message, "result", None)
                structured = getattr(message, "structured_output", None)
                if getattr(message, "is_error", False):
                    raise RuntimeError(
                        "ambient provider failed during a structured-output request: "
                        + _error_detail(message, result_text)
                    )

        if isinstance(structured, dict) and structured:
            return structured
        raw = result_text if result_text else "".join(text_parts)
        return _extract_json_object(raw)


def _error_detail(message: Any, result_text: str | None) -> str:
    """The most informative NON-EMPTY detail for an errored ambient result. ``message.errors``
    is often None (present but null), which stringified to 'None' — an empty error message is
    worse than none: it reads as a bug in us. Fall through the fields that may carry detail and
    guarantee a non-empty string, so the client always gets something actionable to name."""
    return str(
        getattr(message, "errors", None)
        or getattr(message, "subtype", None)
        or result_text
        or "the run reported an error with no detail"
    )


def _extract_json_object(text: str) -> JSONObject:
    """Pull the first top-level JSON object out of a model response.

    Tolerates a stray code fence or surrounding prose; the caller's schema validation is
    the real guard, so this only has to find the object, not trust it.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    start = s.find("{")
    if start == -1:
        raise RuntimeError("ambient response contained no JSON object")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                obj: JSONObject = json.loads(s[start : i + 1])
                return obj
    raise RuntimeError("ambient response had an unterminated JSON object")
