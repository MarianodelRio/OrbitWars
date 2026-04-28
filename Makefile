PYTHON  := .venv/bin/python
CKPT    ?= runs/rl_phase3_league/run_001/checkpoints/rl_best_winrate.pt
RUN     ?= runs/rl_phase3_league/run_001
IL_CKPT ?= runs/planet_policy_il_v3/run_001/checkpoints/best.pt

.PHONY: help match tournament submit submit-neural test test-unit test-integration train train-rl \
        data cache train-bg train-phase1 train-phase2 train-phase3 pipeline eval watch test-quick

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
	@echo "  data               Generate IL training data via tournament runner"
	@echo "  cache              Pre-build HDF5 IL sample cache (data/cache/il_planet_policy.h5)"
	@echo "  train-bg           Launch IL training in background via nohup; prints PID"
	@echo "  train-phase1       Launch RL phase 1 in background (requires IL_CKPT)"
	@echo "  train-phase2       Launch RL phase 2 in background"
	@echo "  train-phase3       Launch RL phase 3 in background"
	@echo "  pipeline           Run full pipeline: data → IL → phase1 → phase2 → phase3 (blocking)"
	@echo "  eval               Evaluate CKPT checkpoint (default: rl_best_winrate.pt)"
	@echo "  watch              Stream live training metrics for RUN directory"
	@echo "  test-quick         Run test suite fast (-x -q); Unix only"

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

data:
	$(PYTHON) scripts/tournament/run.py

cache:
	$(PYTHON) -c "\
from dataset.catalog import DataCatalog; \
from bots.neural.state_builder import StateBuilder; \
from bots.neural.action_codec import ActionCodec; \
from bots.neural.training import build_il_cache; \
import json, pathlib; \
cfg = json.loads(pathlib.Path('training/il_config.json').read_text()); \
bcfg = cfg.get('dataset_builder', cfg); \
mc = cfg.get('model_config', {}); \
sb = StateBuilder(max_planets=mc.get('max_planets', 50), max_fleets=mc.get('max_fleets', 200)); \
codec = ActionCodec(n_amount_bins=mc.get('n_amount_bins', 8)); \
cache_path = pathlib.Path(bcfg.get('cache_path', 'data/cache/il_planet_policy.h5')); \
cache_path.parent.mkdir(parents=True, exist_ok=True); \
build_il_cache(DataCatalog.scan(), sb, codec, cache_path, perspective='both'); \
print('Cache written to', cache_path)"

train-bg:
	nohup $(PYTHON) train.py --config training/il_config.json \
	    > runs/il_train.log 2>&1 & echo "PID: $$!"

train-phase1:
	@test -f "$(IL_CKPT)" || { echo "ERROR: IL checkpoint not found: $(IL_CKPT)"; echo "Run 'make train-bg' first or set IL_CKPT=<path>"; exit 1; }
	nohup $(PYTHON) train.py --config training/rl_phase1.json \
	    > runs/rl_phase1.log 2>&1 & echo "PID: $$!"

train-phase2:
	nohup $(PYTHON) train.py --config training/rl_phase2.json \
	    > runs/rl_phase2.log 2>&1 & echo "PID: $$!"

train-phase3:
	nohup $(PYTHON) train.py --config training/rl_phase3.json \
	    > runs/rl_phase3.log 2>&1 & echo "PID: $$!"

pipeline:
	$(PYTHON) scripts/tournament/run.py
	$(PYTHON) train.py --config training/il_config.json
	$(PYTHON) train.py --config training/rl_phase1.json
	$(PYTHON) train.py --config training/rl_phase2.json
	$(PYTHON) train.py --config training/rl_phase3.json

eval:
	$(PYTHON) train.py eval \
	    --checkpoint $(CKPT) \
	    --opponents heuristic.baseline heuristic.sniper heuristic.oracle_sniper \
	    --n-matches 30

watch:
	@bash scripts/watch_run.sh $(RUN)

test-quick:
	$(PYTHON) -m pytest tests/ -x -q
