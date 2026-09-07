# One-command entry points. `make setup && make run` gives a working, credential-free
# endpoint (replay provider) on a clean machine.

.PHONY: setup data run test lint format typecheck eval check

setup:            ## install Python 3.12 + deps into a local venv (uv handles both)
	uv sync

data:             ## download raw UCI (the synthetic slice is already committed)
	uv run python scripts/fetch_datasets.py

run:              ## serve the API on :8000 with the replay provider (no credentials)
	uv run nl-insights serve --replay

test:             ## run the test suite (hermetic — replay fixtures, no network)
	uv run pytest

lint:             ## static lint
	uv run ruff check .

format:           ## check formatting
	uv run ruff format --check .

typecheck:        ## strict type checking
	uv run mypy

eval:             ## run the eval harness (anti-hardcoding lint + a golden confusion matrix)
	uv run python -m nl_insights.eval

check: lint format typecheck eval test   ## everything CI runs
