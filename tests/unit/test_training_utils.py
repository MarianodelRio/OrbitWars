"""Unit tests for MetricsLogger, RunConfig, and CheckpointManager."""

import csv
import json

import pytest
import torch

from training.utils.metrics import MetricsLogger
from training.utils.run_config import RunConfig
from training.utils.checkpointing import CheckpointManager
from bots.neural.planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    cfg = PlanetPolicyConfig(max_planets=2, max_fleets=4)
    return PlanetPolicyModel(cfg)


def _make_state_builder():
    return StateBuilder(max_planets=2, max_fleets=4)


def _make_codec():
    return ActionCodec(n_amount_bins=5)


def _make_ckpt_manager(run_dir):
    return CheckpointManager(run_dir)


def _minimal_run_config(**overrides):
    defaults = dict(
        run_name="test_run",
        run_id="run_001",
        model_config={"model_type": "planet_policy", "max_planets": 2},
        lr=1e-3,
        batch_size=32,
        epochs=10,
        val_split=0.1,
        eval_every=5,
        eval_opponents=[],
        n_eval_matches=3,
        data_pipeline={},
        device="cpu",
        seed=42,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------

def test_metrics_logger_first_call_writes_header_and_row(tmp_path):
    logger = MetricsLogger(tmp_path / "m.csv", ["epoch", "loss"])
    logger.log({"epoch": 1, "loss": 0.5})
    with open(tmp_path / "m.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0] == {"epoch": "1", "loss": "0.5"}


def test_metrics_logger_second_call_appends_no_duplicate_header(tmp_path):
    logger = MetricsLogger(tmp_path / "m.csv", ["epoch", "loss"])
    logger.log({"epoch": 1, "loss": 0.5})
    logger.log({"epoch": 2, "loss": 0.3})
    with open(tmp_path / "m.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_metrics_logger_csv_readable_with_dictreader(tmp_path):
    logger = MetricsLogger(tmp_path / "m.csv", ["epoch", "loss"])
    for i in range(1, 4):
        logger.log({"epoch": i, "loss": 0.1 * i})
    with open(tmp_path / "m.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[2]["epoch"] == "3"


def test_metrics_logger_append_to_existing_file(tmp_path):
    path = tmp_path / "m.csv"
    logger1 = MetricsLogger(path, ["epoch", "loss"])
    logger1.log({"epoch": 1, "loss": 0.5})

    logger2 = MetricsLogger(path, ["epoch", "loss"])
    logger2.log({"epoch": 2, "loss": 0.4})

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------

def test_next_run_id_empty_directory(tmp_path):
    result = RunConfig._next_run_id(tmp_path / "nonexistent")
    assert result == "run_001"


def test_next_run_id_existing_runs(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "run_001").mkdir()
    (run_dir / "run_002").mkdir()
    result = RunConfig._next_run_id(run_dir)
    assert result == "run_003"


def test_run_config_save_writes_json(tmp_path):
    config = _minimal_run_config()
    config.save(tmp_path)
    config_file = tmp_path / "config.json"
    assert config_file.exists()
    with open(config_file) as f:
        data = json.load(f)
    assert data["run_name"] == "test_run"
    assert "run_id" in data


def test_run_config_run_dir_contains_run_name_and_id():
    config = _minimal_run_config(run_name="my_run", run_id="run_007")
    assert str(config.run_dir).endswith("my_run/run_007")


def test_run_config_weight_decay_defaults_to_1e_minus_4():
    config = _minimal_run_config()
    assert config.weight_decay == pytest.approx(1e-4)


def test_run_config_action_type_loss_weight_defaults_to_1():
    config = _minimal_run_config()
    assert config.action_type_loss_weight == pytest.approx(1.0)


def test_run_config_value_loss_weight_defaults_to_0_5():
    config = _minimal_run_config()
    assert config.value_loss_weight == pytest.approx(0.5)


def test_run_config_use_class_weights_defaults_to_true():
    config = _minimal_run_config()
    assert config.use_class_weights is True


def test_run_config_new_fields_round_trip_json(tmp_path):
    """weight_decay, action_type_loss_weight, value_loss_weight, use_class_weights serialise correctly."""
    config = _minimal_run_config(
        weight_decay=1e-3,
        action_type_loss_weight=2.0,
        value_loss_weight=0.25,
        use_class_weights=False,
    )
    config.save(tmp_path)
    with open(tmp_path / "config.json") as f:
        data = json.load(f)
    assert data["weight_decay"] == pytest.approx(1e-3)
    assert data["action_type_loss_weight"] == pytest.approx(2.0)
    assert data["value_loss_weight"] == pytest.approx(0.25)
    assert data["use_class_weights"] is False


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

def test_checkpoint_save_no_best_creates_epoch_and_last(tmp_path):
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = _make_state_builder()
    codec = _make_codec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=False)
    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "epoch_001.pt").exists()
    assert (ckpt_dir / "last.pt").exists()
    assert not (ckpt_dir / "best.pt").exists()


def test_checkpoint_save_best_creates_three_files(tmp_path):
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = _make_state_builder()
    codec = _make_codec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=True)
    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "epoch_001.pt").exists()
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()


def test_checkpoint_contains_expected_keys(tmp_path):
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = _make_state_builder()
    codec = _make_codec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=False)
    ckpt = torch.load(tmp_path / "checkpoints" / "epoch_001.pt", map_location="cpu", weights_only=False)
    expected_keys = {"config", "state_dict", "epoch", "train_loss", "val_loss", "timestamp",
                     "max_planets", "max_fleets", "n_amount_bins", "model_type"}
    assert expected_keys.issubset(set(ckpt.keys()))


def test_checkpoint_planet_policy_model_type(tmp_path):
    """PlanetPolicyModel checkpoint has model_type == 'planet_policy'."""
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = _make_state_builder()
    codec = _make_codec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=False)
    ckpt = torch.load(tmp_path / "checkpoints" / "epoch_001.pt", map_location="cpu", weights_only=False)
    assert ckpt["model_type"] == "planet_policy"


def test_list_checkpoints_excludes_best_and_last(tmp_path):
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = _make_state_builder()
    codec = _make_codec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=False)
    mgr.save(model, sb, codec, epoch=2, metrics={"train_loss": 0.4, "val_loss": 0.3}, is_best=True)
    paths = mgr.list_checkpoints()
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert "best.pt" not in names
    assert "last.pt" not in names


# ---------------------------------------------------------------------------
# RunConfig.resume_from
# ---------------------------------------------------------------------------

def test_run_config_resume_from_defaults_to_none():
    config = _minimal_run_config()
    assert config.resume_from is None


def test_run_config_resume_from_accepts_path_string():
    config = _minimal_run_config(resume_from="runs/neural_il/run_001/checkpoints/best.pt")
    assert config.resume_from == "runs/neural_il/run_001/checkpoints/best.pt"


def test_run_config_save_round_trips_resume_from(tmp_path):
    config = _minimal_run_config(resume_from="runs/neural_il/run_001/checkpoints/best.pt")
    config.save(tmp_path)
    with open(tmp_path / "config.json") as f:
        data = json.load(f)
    assert data["resume_from"] == "runs/neural_il/run_001/checkpoints/best.pt"


# ---------------------------------------------------------------------------
# load_agent checkpoint syntax
# ---------------------------------------------------------------------------

def test_load_agent_plain_spec_returns_callable():
    from game.env.evaluator import load_agent
    agent = load_agent("bots.heuristic.baseline:agent_fn")
    assert callable(agent)


def test_load_agent_checkpoint_spec_returns_callable(tmp_path):
    """load_agent with ?checkpoint= returns a lazy closure (not yet executed)."""
    from game.env.evaluator import load_agent
    # Point to a non-existent checkpoint — the closure should be created without
    # loading the file (lazy init), so no error at construction time.
    spec = f"bots.neural.bot:agent_fn?checkpoint={tmp_path}/fake.pt"
    agent = load_agent(spec)
    assert callable(agent)


def test_load_agent_checkpoint_name_reflects_file(tmp_path):
    from game.env.evaluator import load_agent
    spec = f"bots.neural.bot:agent_fn?checkpoint={tmp_path}/best.pt"
    agent = load_agent(spec)
    assert "best.pt" in agent.__name__


def test_load_agent_checkpoint_loads_on_first_call(tmp_path):
    """End-to-end: save a real checkpoint and verify load_agent can load and call it."""
    from game.env.evaluator import load_agent

    cfg = PlanetPolicyConfig(max_planets=2, max_fleets=4)
    model = PlanetPolicyModel(cfg)
    sb = StateBuilder(max_planets=2, max_fleets=4)
    codec = ActionCodec(n_amount_bins=5)
    mgr = _make_ckpt_manager(tmp_path)
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=True)
    ckpt_path = str(tmp_path / "checkpoints" / "best.pt")

    spec = f"bots.neural.bot:agent_fn?checkpoint={ckpt_path}"
    agent = load_agent(spec)

    obs = {
        "player": 0,
        "planets": [[0, 0, 10.0, 20.0, 1.0, 50.0, 3.0], [1, 1, 80.0, 70.0, 1.0, 30.0, 2.0]],
        "fleets": [],
        "comet_planet_ids": [],
        "step": 0,
    }
    result = agent(obs)
    assert isinstance(result, list)
