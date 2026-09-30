.PHONY: install doctor test lint notebooks verify verify-live lab api eval
install:
	uv sync --locked

doctor:
	uv run python scripts/doctor.py

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests clases

notebooks:
	uv run python scripts/sync_notebooks.py

verify:
	uv run python scripts/verify.py --mode offline

verify-live:
	uv run python scripts/verify.py --mode live

lab:
	uv run jupyter lab clases

api:
	uv run uvicorn henry_agents.api:app --host 127.0.0.1 --port 8000

eval:
	uv run python scripts/evaluate.py --mode offline
