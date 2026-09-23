UV_CACHE_DIR ?= .cache/uv
export UV_CACHE_DIR

.PHONY: setup fmt lint typecheck test check mock-validate

setup:
	uv sync --dev

fmt:
	uv run --dev ruff format src tests
	uv run --dev ruff check --fix src tests

lint:
	uv run --dev ruff check src tests
	uv run --dev ruff format --check src tests

typecheck:
	uv run --dev mypy --strict src

test:
	uv run --dev pytest -q

check: lint typecheck test
	cmp CLAUDE.md AGENTS.md

mock-validate:
	uv run python -m llmbench.loadtest.validation --output-dir results/validation
