UV = uv run

install:
	uv sync

ruff:
	$(UV) ruff check src --fix && $(UV) ruff format src

ruff-ci:
	$(UV) ruff check src && $(UV) ruff format --check src

format: ruff

check: ruff-ci

evaluate:
	$(UV) python -m wmt26_terminology.evaluate --submissions $(SUBMISSIONS) --out $(OUT)

evaluate-gold:
	$(UV) python -m wmt26_terminology.evaluate --gold

.PHONY: install evaluate evaluate-gold ruff ruff-ci format check
