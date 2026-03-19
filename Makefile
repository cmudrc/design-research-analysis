PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(shell if command -v python3.12 >/dev/null 2>&1; then echo python3.12; else echo python3; fi))
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
SPHINX ?= $(PYTHON) -m sphinx
BUILD ?= $(PYTHON) -m build
TWINE ?= $(PYTHON) -m twine
RUNTIME_CACHE_DIR ?= artifacts/runtime
MPLCONFIGDIR ?= $(RUNTIME_CACHE_DIR)/matplotlib
XDG_CACHE_HOME ?= $(RUNTIME_CACHE_DIR)/xdg-cache
MPLBACKEND ?= Agg

export MPLCONFIGDIR
export XDG_CACHE_HOME
export MPLBACKEND

.PHONY: help check-python dev install-dev \
	lint fmt fmt-check type test qa coverage docstrings-check \
	runtime-cache run-example run-examples examples-coverage docs docs-build docs-check docs-linkcheck \
	release-check ci clean

help:
	@echo "Common targets:"
	@echo "  dev              Install the project in editable mode with dev dependencies."
	@echo "  test             Run the pytest suite."
	@echo "  qa               Run lint, fmt-check, type, and test."
	@echo "  run-example      Execute the bundled example script."
	@echo "  run-examples     Execute all example scripts."
	@echo "  examples-coverage Check public API coverage across examples."
	@echo "  docs             Build the HTML docs."
	@echo "  ci               Run the main local CI checks."

check-python:
	@$(PYTHON) -c "import pathlib, sys; print(f'Using Python {sys.version.split()[0]} at {pathlib.Path(sys.executable)}'); raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" || (echo "Python >= 3.12 is required by pyproject.toml"; exit 1)

runtime-cache:
	mkdir -p "$(MPLCONFIGDIR)" "$(XDG_CACHE_HOME)"

dev:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

install-dev: dev

lint: check-python
	$(RUFF) check .

fmt: check-python
	$(RUFF) format .

fmt-check: check-python
	$(RUFF) format --check .

type: check-python
	$(MYPY) src

test: check-python runtime-cache
	PYTHONPATH=src $(PYTEST) -q

qa: lint fmt-check type test

coverage: check-python runtime-cache
	mkdir -p artifacts/coverage
	PYTHONPATH=src $(PYTEST) --cov=src/design_research_analysis --cov-report=term --cov-report=json:artifacts/coverage/coverage.json -q
	$(PYTHON) scripts/check_coverage_thresholds.py --coverage-json artifacts/coverage/coverage.json --minimum 90

docstrings-check: check-python
	$(PYTHON) scripts/check_google_docstrings.py

run-example: check-python runtime-cache
	PYTHONPATH=src $(PYTHON) examples/basic_usage.py

run-examples: check-python runtime-cache
	@set -e; \
	for script in $$(ls examples/*.py | sort); do \
		echo "Running $$script"; \
		PYTHONPATH=src $(PYTHON) "$$script"; \
	done

examples-coverage: check-python
	$(PYTHON) scripts/check_example_api_coverage.py --minimum 35

docs-build: check-python runtime-cache
	$(PYTHON) scripts/generate_example_docs.py
	PYTHONPATH=src $(SPHINX) -b html docs docs/_build/html -n -W --keep-going -E

docs-check: check-python
	$(PYTHON) scripts/generate_example_docs.py --check
	$(PYTHON) scripts/check_docs_consistency.py

docs-linkcheck: check-python runtime-cache
	PYTHONPATH=src $(SPHINX) -b linkcheck docs docs/_build/linkcheck -W --keep-going -E

docs: docs-build

release-check: check-python
	rm -rf build dist
	$(BUILD)
	$(TWINE) check dist/*

ci: qa coverage docstrings-check docs-check run-examples examples-coverage release-check

clean:
	rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache artifacts build dist docs/_build
	find src -maxdepth 2 -type d -name "*.egg-info" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name ".coverage.*" \) -exec rm -f {} + 2>/dev/null || true
