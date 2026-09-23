UV_CACHE_DIR ?= .cache/uv
export UV_CACHE_DIR

.PHONY: setup fmt lint typecheck test check mock-validate

setup:
	uv sync --dev

fmt:
	uv run --dev ruff format src tests modal_app
	uv run --dev ruff check --fix src tests modal_app

lint:
	uv run --dev ruff check src tests modal_app
	uv run --dev ruff format --check src tests modal_app

typecheck:
	uv run --dev mypy --strict src

test:
	uv run --dev pytest -q

check: lint typecheck test
	cmp CLAUDE.md AGENTS.md

mock-validate:
	uv run python -m llmbench.loadtest.validation --output-dir results/validation

# --- Docker (local, $0) ---
.PHONY: docker-check
docker-check:
	hadolint docker/Dockerfile
	docker compose config --quiet

# --- Modal (BILLABLE: print an estimate and get explicit approval first) ---
# All runs are detached, so a dropped client never kills a paid job.
MODAL_RUN = uv run modal run --detach -m
SMOKE_CONFIG ?= phase3_smoke.yaml

.PHONY: modal-download modal-checks smoke smoke-vllm smoke-hf sync-results
modal-download:  # CPU only
	$(MODAL_RUN) modal_app.download --config $(SMOKE_CONFIG)

modal-checks:  # CPU only: builds both images, mock validation inside vLLM image
	$(MODAL_RUN) modal_app.checks::vllm_env --config $(SMOKE_CONFIG)
	$(MODAL_RUN) modal_app.checks::hf_env

smoke-vllm:  # L4
	$(MODAL_RUN) modal_app.smoke::vllm --config $(SMOKE_CONFIG)

smoke-hf:  # L4
	$(MODAL_RUN) modal_app.smoke::hf --config $(SMOKE_CONFIG)

smoke: smoke-vllm smoke-hf

sync-results:  # $0: copy Phase 3 results from the results Volume
	uv run modal volume get llmbench-results phase3 results/validation --force
