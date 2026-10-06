.PHONY: desde-notebooks setup install doctor doctor-live test lint notebooks verify verify-live lab api eval verify-workflows eval-workflows
# En Windows sin make, usar los comandos de cada regla directamente (ver docs/INSTALACION.md).
setup:
	uv sync --locked
	uv run python scripts/doctor.py

install:
	uv sync --locked

doctor:
	uv run python scripts/doctor.py

doctor-live:
	uv run python scripts/doctor.py --live

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests clases proyectos

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

desde-notebooks:
	uv run python scripts/sync_notebooks.py --desde-notebooks

verify-workflows:
	uv run python scripts/verify.py --mode offline --track workflows

eval-workflows:
	uv run python scripts/evaluate_workflows.py
