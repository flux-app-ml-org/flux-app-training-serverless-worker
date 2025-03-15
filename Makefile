# Makefile

# Define the test command
TEST_CMD = python3 -m pytest -xvs tests/ -p no:cov

# Default target
all: test

# Target to run tests
test:
	$(TEST_CMD)

# Clean target (optional, if you want to add cleaning functionality)
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
