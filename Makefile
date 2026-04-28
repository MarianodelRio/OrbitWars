PYTHON := .venv/bin/python

.PHONY: help match tournament submit submit-neural test test-unit test-integration train train-rl

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  match              Run a single match (config: scripts/matches/config.json)"
	@echo "  tournament         Run round-robin tournament with ELO (config: scripts/tournament/config.json)"
	@echo "  submit             Package bot and submit to Kaggle (config: scripts/submission/config.json)"
	@echo "  submit-neural      Package neural bot (flat/pointer) and submit"
	@echo "  test               Run all tests"
	@echo "  test-unit          Run unit tests only"
	@echo "  test-integration   Run integration tests only"
	@echo "  train              Run IL training — flat MLP (config: training/il_config.json)"
	@echo "  train-rl           Run RL/PPO training (config: CONFIG=path/to/rl_config.json)"

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

test-unit:
	$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v

train:
	$(PYTHON) scripts/train_il.py --config training/il_config.json

train-rl:
	$(PYTHON) scripts/train_rl.py --config $(CONFIG)
