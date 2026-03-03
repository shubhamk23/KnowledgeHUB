# ──────────────────────────────────────────────────────────────────────────────
# AI Notes Knowledge Hub — Developer Makefile
# ──────────────────────────────────────────────────────────────────────────────
# Usage:
#   make test           Run the full test suite with coverage
#   make test-unit      Only unit tests (fast, no DB)
#   make test-int       Only integration tests (API against in-memory SQLite)
#   make test-e2e       Only end-to-end journey tests
#   make coverage       Run tests and open HTML coverage report
#   make install        Install all Python dependencies (incl. test deps)
#   make start          Start both backend and frontend
# ──────────────────────────────────────────────────────────────────────────────

BACKEND_DIR := backend
PYTHON      := python3
PIP         := pip3
PYTEST      := pytest

.PHONY: install test test-unit test-int test-e2e coverage start help

# ── Dependency installation ────────────────────────────────────────────────────
install:
	@echo "Installing Python dependencies..."
	cd $(BACKEND_DIR) && $(PIP) install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# ── Test targets ───────────────────────────────────────────────────────────────

## Run every test (unit + integration + e2e) with coverage
test:
	@echo ""
	@echo "========================================"
	@echo "  Running full test suite with coverage"
	@echo "========================================"
	cd $(BACKEND_DIR) && $(PYTEST) --cov=app --cov-report=term-missing --cov-report=html:htmlcov -v

## Run only unit tests (pure-function tests — no I/O)
test-unit:
	@echo ""
	@echo "========================================"
	@echo "  Running UNIT tests"
	@echo "========================================"
	cd $(BACKEND_DIR) && $(PYTEST) -m unit -v --no-cov

## Run only integration tests (API endpoints against in-memory SQLite)
test-int:
	@echo ""
	@echo "========================================"
	@echo "  Running INTEGRATION tests"
	@echo "========================================"
	cd $(BACKEND_DIR) && $(PYTEST) -m integration -v --no-cov

## Run only end-to-end journey tests
test-e2e:
	@echo ""
	@echo "========================================"
	@echo "  Running END-TO-END tests"
	@echo "========================================"
	cd $(BACKEND_DIR) && $(PYTEST) -m e2e -v --no-cov

## Run tests and open HTML coverage report in the browser
coverage: test
	@echo ""
	@echo "Opening HTML coverage report..."
	open $(BACKEND_DIR)/htmlcov/index.html || xdg-open $(BACKEND_DIR)/htmlcov/index.html || echo "Open backend/htmlcov/index.html manually"

## Run tests in watch mode (requires pytest-watch: pip install pytest-watch)
test-watch:
	cd $(BACKEND_DIR) && ptw -- -m "unit or integration" -v --no-cov

# ── Start services ─────────────────────────────────────────────────────────────
start:
	@echo "Starting AI Notes Knowledge Hub..."
	bash start.sh

# ── Help ───────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "AI Notes Knowledge Hub — Make Targets"
	@echo "────────────────────────────────────────────────────────────"
	@echo "  make install      Install Python + npm dependencies"
	@echo "  make test         Full test suite (unit + int + e2e + coverage)"
	@echo "  make test-unit    Unit tests only (fast)"
	@echo "  make test-int     Integration tests only"
	@echo "  make test-e2e     End-to-end journey tests only"
	@echo "  make coverage     Run tests + open HTML coverage report"
	@echo "  make start        Start backend (port 8000) + frontend (port 3000)"
	@echo ""
