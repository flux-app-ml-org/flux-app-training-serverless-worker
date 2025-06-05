# Makefile

# Define variables
VENV_DIR = venv
PYTHON = python3
PIP = $(VENV_DIR)/bin/pip3
PYTHON_VENV = $(VENV_DIR)/bin/python3
TEST_CMD = $(PYTHON_VENV) -m pytest -xvs tests/ -p no:cov

# Default target
all: test

# Install system dependencies (requires sudo)
install-system-deps:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y libsqlite3-dev libffi-dev libbz2-dev libncurses-dev \
	                       libreadline-dev libssl-dev zlib1g-dev libgdbm-dev \
	                       liblzma-dev tk-dev

# Setup virtual environment
setup-venv:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

# Install Python dependencies
install-deps: setup-venv
	@echo "Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

# Full setup (system deps + venv + python deps)
setup: install-system-deps install-deps
	@echo "Setup complete!"

# Target to run tests (depends on setup)
test: install-deps
	@echo "Running tests..."
	$(TEST_CMD)

# Quick test (assumes setup is already done)
test-quick:
	@echo "Running tests (quick - no setup)..."
	$(TEST_CMD)

# Clean target
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	rm -rf $(VENV_DIR)

# Help target
help:
	@echo "Available targets:"
	@echo "  setup              - Full setup (system deps + venv + python deps)"
	@echo "  install-system-deps - Install system dependencies (requires sudo)"
	@echo "  setup-venv         - Create virtual environment"
	@echo "  install-deps       - Install Python dependencies"
	@echo "  test               - Run tests (includes dependency setup)"
	@echo "  test-quick         - Run tests without setup"
	@echo "  clean              - Clean up __pycache__ and venv"
	@echo "  help               - Show this help message"

.PHONY: all setup install-system-deps setup-venv install-deps test test-quick clean help
