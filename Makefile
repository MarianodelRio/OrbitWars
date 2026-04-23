PYTHON := .venv/bin/python

.PHONY: help match tournament submit submit-neural test train

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  match       Run a single match (config: scripts/matches/config.json)"
	@echo "  tournament  Run round-robin tournament with ELO (config: scripts/tournament/config.json)"
	@echo "  submit        Package bot and submit to Kaggle (config: scripts/submission/config.json)"
	@echo "  submit-neural  Package neural bot and submit (config: scripts/submission/neural_config.json)"
	@echo "  test          Run all tests"
	@echo "  train       Run imitation learning training (config: training/il_config.json)"

match:
	$(PYTHON) scripts/matches/run.py

tournament:
	$(PYTHON) scripts/tournament/run.py

submit:
	@set -a && . ./.env && set +a && $(PYTHON) scripts/submission/run.py

submit-neural:
	@set -a && . ./.env && set +a && $(PYTHON) scripts/submission/package_neural.py

test:
	$(PYTHON) -m pytest tests/ -v

train:
	$(PYTHON) scripts/train_il.py --config training/il_config.json
