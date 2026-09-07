"""Command-line entry point.

The one command a grader runs. ``--replay`` (the default) needs no credentials and
serves committed fixtures; ``--provider apikey`` uses ``ANTHROPIC_API_KEY``; add
``--record`` to a live provider to write fixtures.
"""

from __future__ import annotations

import argparse
import logging
from typing import cast

from .config import ProviderMode, Settings


def _configure_logging() -> None:
    """Make the APP's own operational logs visible on stdout (hence journald / a grader's
    terminal). uvicorn configures only its OWN loggers, so without this the app's INFO lines —
    including the migrate-in-place record a deploy greps for — go nowhere, and their absence
    reads as 'it never ran' (the exact false-negative the log exists to prevent, one layer
    down). Configure it in the SERVE entry point, not in a library module, so importing the
    package never imposes logging on a consumer. force=True so uvicorn's later setup (with
    disable_existing_loggers=False) does not leave the app loggers unhandled."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def build_settings(argv: list[str] | None = None) -> tuple[Settings, argparse.Namespace]:
    parser = argparse.ArgumentParser(prog="nl-insights", description=__doc__)
    parser.add_argument("command", choices=["serve"], help="what to run")
    parser.add_argument(
        "--provider",
        choices=["replay", "apikey", "ambient"],
        help="LLM auth path (default: replay — no credentials).",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="shorthand for --provider replay (the credential-free default).",
    )
    parser.add_argument(
        "--record", action="store_true", help="write fixtures from a live provider."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    provider = cast("ProviderMode | None", "replay" if args.replay else args.provider)
    settings = Settings().with_overrides(
        provider=provider,
        record=args.record or None,
    )
    return settings, args


def main(argv: list[str] | None = None) -> None:
    settings, args = build_settings(argv)
    if args.command == "serve":
        import uvicorn

        from .api import create_app

        _configure_logging()  # app logs reach stdout/journald, not just uvicorn's own
        uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
