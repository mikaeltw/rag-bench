# ---- Config ----
PY ?= 3.12
UV ?= uv
TOX_ENVS ?= py311,py312,py313

# Default tests run on CPU for stability
export RAG_BENCH_DEVICE ?= cpu
export CUDA_VISIBLE_DEVICES ?=

# ---- Phony targets ----
.PHONY: help setup sync dev lint typecheck format test test-all test-py build clean distclean \
        coverage coverage-xml coverage-report coverage-erase coveralls-upload-local

help:
	@echo "Common tasks:"
	@echo "  make setup          Install uv (if needed)"
	@echo "  make sync           Create/refresh local venv with dev deps"
	@echo "  make dev            Lint + typecheck + unit/offline tests"
	@echo "  make lint           flake8 + isort --check + black --check"
	@echo "  make typecheck      mypy over src/"
	@echo "  make format         Apply isort + black"
	@echo "  make test           Unit/offline tests on current Python"
	@echo "  make test-all       Matrix tests via tox (py311/12/13)"
	@echo "  make build          Build sdist + wheel"
	@echo "  make coverage       Run tests and produce combined coverage report"
	@echo "  make coveralls-upload-local  Upload coverage.xml from local machine to Coveralls"
	@echo "  make clean          Remove caches/build artefacts"
	@echo "  make distclean      Also remove venvs and tox envs"

setup:
	@command -v $(UV) >/dev/null || (echo "Installing uv..."; \
		curl -fsSL https://astral.sh/uv/install.sh | sh)
	@echo "uv installed: $$($(UV) --version)"

sync:
	$(UV) python install $(PY)
	$(UV) venv
	$(UV) sync --all-extras --dev

dev: sync lint typecheck test

lint:
	$(UV) run flake8
	$(UV) run isort --check-only .
	$(UV) run black --check .

typecheck:
	$(UV) run mypy src/rag_bench

format:
	$(UV) run isort .
	$(UV) run black .

test:
	$(UV) run pytest -q -m "unit or offline" --disable-warnings

test-all:
	$(UV) tool run --from tox-uv tox -e $(TOX_ENVS)

test-py:
	$(UV) python install $(PY)
	PYTHON=$(PY) $(UV) run pytest -q -m "unit or offline" --disable-warnings

build:
	$(UV) run python -m build

# -------- Coverage & Coveralls --------

coverage-erase:
	$(UV) run coverage erase

coverage:
	# Run your local default slice with coverage; add more slices if you like.
	$(UV) run coverage erase
	$(UV) run pytest -q -m "unit or offline" --cov=rag_bench --cov-branch --cov-report=
	$(UV) run coverage combine
	$(UV) run coverage report
	@echo "Use 'make coverage-xml' for XML (Coveralls) or 'make coveralls-upload-local' to submit from local machine."

coverage-xml:
	$(UV) run coverage xml
	@echo "Wrote coverage.xml"

coverage-report:
	$(UV) run coverage report

clean:
	@rm -rf .pytest_cache .mypy_cache dist build coverage.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} +

distclean: clean
	@rm -rf .venv .tox
