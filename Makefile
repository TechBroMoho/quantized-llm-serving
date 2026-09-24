UV_CACHE_DIR ?= .cache/uv
export UV_CACHE_DIR

.PHONY: setup fmt lint typecheck test check mock-validate

setup:
	uv sync --dev

fmt:
	uv run --dev ruff format src tests modal_app scripts
	uv run --dev ruff check --fix src tests modal_app scripts

lint:
	uv run --dev ruff check src tests modal_app scripts
	uv run --dev ruff format --check src tests modal_app scripts

typecheck:
	uv run --dev mypy --strict src

test:
	uv run --dev pytest -q

check: lint typecheck test
	cmp CLAUDE.md AGENTS.md

MOCK_VALIDATE_DIR ?= results/validation
mock-validate:  # $0: timing (1 and 2 client processes) + 2-process capacity gate
	uv run python -m llmbench.loadtest.validation --output-dir $(MOCK_VALIDATE_DIR)

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

# --- Phase 6 benchmarks (BILLABLE except the rehearsal and sync) ---
BENCH_RUN = $(MODAL_RUN) modal_app.bench
.PHONY: bench-prompts-rehearsal bench-prepare bench sync-bench
bench-prompts-rehearsal: eval-env  # $0: prompt build on WikiText-103 validation
	HF_HOME=$(CURDIR)/.cache/hf-eval PYTHONPATH=src $(EVAL_VENV)/bin/python \
		scripts/bench_prompts_rehearsal.py $(QWEN3_TOKENIZER) \
		results/validation/phase6/prompts_rehearsal.json

bench-prepare:  # CPU only: checkpoint sha256 check + WikiText-103 prompt pool
	$(BENCH_RUN)::prepare

DROP ?=
bench:  # L40S: LIFETIME=<lifetime in configs/phase6_bench.yaml>, DROP=labels (ask first)
	$(BENCH_RUN)::run --lifetime $(LIFETIME) --drop "$(DROP)"

sync-bench:  # $0: copy Phase 6 results (raw request files stay gzipped)
	uv run modal volume get llmbench-results phase6 results/perf --force

# --- Phase 6 prerequisites ---
.PHONY: modal-loadtest-check sync-loadtest-check
modal-loadtest-check:  # CPU only (BILLABLE, tiny): ADR-013 gate in the vLLM image
	$(MODAL_RUN) modal_app.checks::loadtest

sync-loadtest-check:  # $0: copy the in-container load tester checks
	uv run modal volume get llmbench-results phase6 results/validation --force

# --- Phase 4 quantization ---
QUANT_VENV = .cache/quant-venv
.PHONY: quantize-env quantize-rehearsal
quantize-env:  # isolated env: llm-compressor needs transformers 4.55.2
	uv venv -q --allow-existing --python 3.12 $(QUANT_VENV)
	VIRTUAL_ENV=$(QUANT_VENV) uv pip install -q -r requirements/quantize.in

quantize-rehearsal: quantize-env  # $0: CPU run of the exact quantization code
	PYTHONPATH=src $(QUANT_VENV)/bin/python scripts/quantize_rehearsal.py

# --- Phase 5 accuracy ---
EVAL_VENV = .cache/eval-venv
.PHONY: eval-env eval-rehearsal eval-audit eval-prefetch eval-probe eval-full sync-accuracy accuracy-table
eval-env:  # isolated env: lm-eval 0.4.11 with the vLLM image's transformers
	uv venv -q --allow-existing --python 3.12 $(EVAL_VENV)
	VIRTUAL_ENV=$(EVAL_VENV) uv pip install -q -r requirements/eval.in \
		torch==2.8.0 transformers==4.56.1 tokenizers==0.22.0 huggingface-hub==0.34.4 \
		fsspec==2025.9.0 dill==0.4.0 accelerate==1.10.0 pyyaml==6.0.2

eval-rehearsal: eval-env  # $0: CPU run of the exact evaluation code (hf backend)
	HF_HOME=$(CURDIR)/.cache/hf-eval PYTHONPATH=src $(EVAL_VENV)/bin/python scripts/eval_rehearsal.py

QWEN3_REVISION = b968826d9c46dd6066d109eabc6255188de91218
QWEN3_TOKENIZER = .cache/hf-eval/hub/models--Qwen--Qwen3-8B/snapshots/$(QWEN3_REVISION)
eval-audit: eval-env  # $0: token length of every MMLU 5-shot / WikiText request
	HF_HOME=.cache/hf-eval $(EVAL_VENV)/bin/python -c "from huggingface_hub import \
		snapshot_download; snapshot_download('Qwen/Qwen3-8B', revision='$(QWEN3_REVISION)', \
		allow_patterns=['tokenizer*', 'vocab.json', 'merges.txt'])"
	HF_HOME=.cache/hf-eval TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
		$(EVAL_VENV)/bin/python -m llmbench.eval_audit --tokenizer $(QWEN3_TOKENIZER) \
		--max-length 4096 --out results/accuracy/phase5/prompt_lengths_local.json

eval-prefetch:  # CPU only (BILLABLE, tiny): image check, datasets, in-image audit
	$(MODAL_RUN) modal_app.evaluate::prefetch

eval-probe:  # L40S (BILLABLE): one variant with --limit, to time the full run
	$(MODAL_RUN) modal_app.evaluate::probe

eval-full:  # L40S (BILLABLE): MMLU 5-shot + WikiText-2 for BF16, AWQ, GPTQ
	$(MODAL_RUN) modal_app.evaluate::full

sync-accuracy:  # $0: copy Phase 5 results (minus per-sample JSONL) from the Volume
	mkdir -p .cache/phase5-sync
	uv run modal volume get llmbench-results phase5 .cache/phase5-sync --force
	uv run python scripts/sync_accuracy.py

accuracy-table:  # $0: comparison.json + accuracy_table.md from a synced full run
	uv run python -m llmbench.accuracy --run-dir $(RUN_DIR)

# --- Phase 7 analysis ($0, local only) ---
.PHONY: aggregate plots report
aggregate:  # results/ -> results/analysis/aggregate.json (every number + its source)
	uv run python -m llmbench.analysis.aggregate

plots: aggregate  # charts -> results/analysis/*.png
	uv run python -m llmbench.analysis.plots

report: aggregate  # docs/RESULTS.md, every number linked to its raw file
	uv run python -m llmbench.analysis.report
