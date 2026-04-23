"""Unit tests for MetricsLogger, RunConfig, and CheckpointManager."""

import csv
import json

import pytest
import torch

from training.utils.metrics import MetricsLogger
from training.utils.run_config import RunConfig
from training.utils.checkpointing import CheckpointManager
from bots.neural.model import PolicyValueConfig, PolicyValueModel
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model():
    return PolicyValueModel(PolicyValueConfig(input_dim=16, hidden_dims=[32]))


def _make_ckpt_manager(run_dir):
    return CheckpointManager(run_dir)


def _minimal_run_config(**overrides):
    defaults = dict(
        run_name="test_run",
        run_id="run_001",
        model_config={"input_dim": 16, "hidden_dims": [32]},
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


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

def test_checkpoint_save_no_best_creates_epoch_and_last(tmp_path):
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = StateBuilder()
    codec = ActionCodec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=False)
    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "epoch_001.pt").exists()
    assert (ckpt_dir / "last.pt").exists()
    assert not (ckpt_dir / "best.pt").exists()


def test_checkpoint_save_best_creates_three_files(tmp_path):
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = StateBuilder()
    codec = ActionCodec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=True)
    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "epoch_001.pt").exists()
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()


def test_checkpoint_contains_expected_keys(tmp_path):
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = StateBuilder()
    codec = ActionCodec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=False)
    ckpt = torch.load(tmp_path / "checkpoints" / "epoch_001.pt", map_location="cpu")
    expected_keys = {"config", "state_dict", "epoch", "train_loss", "val_loss", "timestamp",
                     "max_planets", "max_fleets", "n_amount_bins"}
    assert expected_keys.issubset(set(ckpt.keys()))


def test_list_checkpoints_excludes_best_and_last(tmp_path):
    mgr = _make_ckpt_manager(tmp_path)
    model = _make_model()
    sb = StateBuilder()
    codec = ActionCodec()
    mgr.save(model, sb, codec, epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4}, is_best=False)
    mgr.save(model, sb, codec, epoch=2, metrics={"train_loss": 0.4, "val_loss": 0.3}, is_best=True)
    paths = mgr.list_checkpoints()
    assert len(paths) == 2
    names = {p.name for p in paths}
    assert "best.pt" not in names
    assert "last.pt" not in names
