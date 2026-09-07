"""HTTP API — exposes ingestion, jobs, and query routes (health only at scaffold stage)."""

from .app import ApiError, create_app

__all__ = ["ApiError", "create_app"]
