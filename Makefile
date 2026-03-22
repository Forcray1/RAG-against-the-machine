PYTHON = python3
MODULE = src.main

install:
	uv sync

run:$
	uv run python -m $(MODULE) $(filter-out $@,$(MAKECMDGOALS))

debug:
	uv run python -m pdb src/main.py

clean:
	rm -rf data/processed/*
	rm -rf data/output/*
	rm -rf .venv
	rm -rf uv.lock
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

%:
	@:

.PHONY: install run debug clean lint lint-strict