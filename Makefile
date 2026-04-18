PYTHON = python3
MODULE = student.__main__

install:
	uv sync

run:$
	uv run python -m $(MODULE) $(filter-out $@,$(MAKECMDGOALS))

run_menu:
	uv run python -m student.UI.menu

debug:
	uv run python -m pdb student/main.py

clean:
	rm -rf data/processed/*
	rm -rf data/output/*
	rm -rf output
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

lint:
	uv run flake8 student
	uv run mypy student --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 student
	uv run mypy student --strict

%:
	@:

.PHONY: install run debug clean lint lint-strict run_menu