"""The serve entry point must make the APP's own logs visible, or a deploy that greps the
journal for the migrate-in-place line reads empty on a perfect migration — the false-negative
the line exists to prevent, relocated one layer down (the log emits, but nowhere a reader sees).
"""

from __future__ import annotations

import logging


def test_serve_logging_puts_app_info_logs_on_the_stream(capsys) -> None:  # type: ignore[no-untyped-def]
    from nl_insights.cli import _configure_logging

    _configure_logging()
    # the exact logger and message shape the migrate-in-place path emits
    logging.getLogger("nl_insights.jobs.store").info(
        "migrated dataset overseer from model v2 to v3: re-derived 5 numeric columns, "
        "3 categorical value sets"
    )
    for h in logging.getLogger().handlers:
        h.flush()
    err = capsys.readouterr().err
    # the acceptance grep 'migrated dataset .* v2 to v3' must find it
    assert "migrated dataset" in err and "v2 to v3" in err
    assert logging.getLogger("nl_insights.jobs.store").isEnabledFor(logging.INFO)
