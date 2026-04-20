PYTHON := .venv/bin/python

.PHONY: help match tournament submit test

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  match       Run a single match (config: scripts/matches/config.json)"
	@echo "  tournament  Run round-robin tournament with ELO (config: scripts/tournament/config.json)"
	@echo "  submit      Package bot and submit to Kaggle (config: scripts/submission/config.json)"
	@echo "  test        Run all tests"

match:
	$(PYTHON) scripts/matches/run.py

tournament:
	$(PYTHON) scripts/tournament/run.py

submit:
	@set -a && . ./.env && set +a && $(PYTHON) scripts/submission/run.py

test:
	$(PYTHON) -m pytest tests/ -v
