"""Unit tests for NeuralILDataset item shapes, dtypes, and values."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch

from bots.neural.training import NeuralILDataset, ILSample
from bots.neural.types import ModelLabels, PerPlanetLabels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_PLANETS = 50
MAX_FLEETS = 100
FLEET_INPUT_DIM = MAX_FLEETS * 7  # 700


def _make_sample(action_type=1, source_idx=0, target_idx=1, amount_bin=2, value_target=0.8):
    return ILSample(
        state_array=np.zeros(1050, dtype=np.float32),
        labels=ModelLabels(
            action_type=action_type,
            source_idx=source_idx,
            target_idx=target_idx,
            amount_bin=amount_bin,
            value_target=value_target,
        ),
    )


def _make_pointer_sample(
    action_type=1,
    source_idx=2,
    target_idx=5,
    amount_bin=3,
    value_target=0.5,
    n_real=10,
):
    """ILSample with pointer-mode arrays populated."""
    planet_features = np.random.randn(MAX_PLANETS, 7).astype(np.float32)
    fleet_features = np.random.randn(FLEET_INPUT_DIM).astype(np.float32)
    planet_mask = np.zeros(MAX_PLANETS, dtype=bool)
    planet_mask[:n_real] = True
    return ILSample(
        state_array=None,
        labels=ModelLabels(
            action_type=action_type,
            source_idx=source_idx,
            target_idx=target_idx,
            amount_bin=amount_bin,
            value_target=value_target,
        ),
        planet_features=planet_features,
        fleet_features=fleet_features,
        planet_mask=planet_mask,
    )


# ---------------------------------------------------------------------------
# Flat-mode tests (unchanged)
# ---------------------------------------------------------------------------

def test_neural_il_dataset_len():
    dataset = NeuralILDataset([_make_sample(), _make_sample()])
    assert len(dataset) == 2


def test_neural_il_dataset_getitem_keys():
    dataset = NeuralILDataset([_make_sample()])
    item = dataset[0]
    assert set(item.keys()) == {"state", "action_type", "source_idx", "target_idx", "amount_bin", "value_target"}


def test_neural_il_dataset_state_shape_and_dtype():
    dataset = NeuralILDataset([_make_sample()])
    item = dataset[0]
    assert item["state"].shape == torch.Size([1050])
    assert item["state"].dtype == torch.float32


def test_neural_il_dataset_label_values():
    dataset = NeuralILDataset([_make_sample(action_type=1, source_idx=3, target_idx=7, amount_bin=2, value_target=0.9)])
    item = dataset[0]
    assert item["action_type"].item() == 1
    assert item["source_idx"].item() == 3
    assert item["target_idx"].item() == 7
    assert item["amount_bin"].item() == 2
    assert item["value_target"].item() == pytest.approx(0.9)


def test_neural_il_dataset_noop_labels():
    dataset = NeuralILDataset([_make_sample(action_type=0, source_idx=-1, target_idx=-1, amount_bin=-1, value_target=0.0)])
    item = dataset[0]
    assert item["action_type"].item() == 0
    assert item["source_idx"].item() == -1
    assert item["target_idx"].item() == -1
    assert item["amount_bin"].item() == -1
    assert item["value_target"].item() == pytest.approx(0.0)


def test_neural_il_dataset_empty():
    dataset = NeuralILDataset([])
    assert len(dataset) == 0


# ---------------------------------------------------------------------------
# Pointer-mode tests
# ---------------------------------------------------------------------------

def test_pointer_mode_flag_false_by_default():
    dataset = NeuralILDataset([_make_sample()])
    assert dataset.use_pointer is False


def test_pointer_mode_flag_true_when_set():
    dataset = NeuralILDataset([_make_pointer_sample()], use_pointer=True)
    assert dataset.use_pointer is True


def test_pointer_mode_keys():
    dataset = NeuralILDataset([_make_pointer_sample()], use_pointer=True)
    item = dataset[0]
    expected_keys = {
        "planet_features", "fleet_features", "planet_mask",
        "action_type", "source_idx", "target_idx", "amount_bin", "value_target",
    }
    assert set(item.keys()) == expected_keys
    assert "state" not in item


def test_pointer_mode_no_state_key():
    """Pointer mode must not include 'state' key."""
    dataset = NeuralILDataset([_make_pointer_sample()], use_pointer=True)
    item = dataset[0]
    assert "state" not in item


def test_pointer_mode_planet_features_shape():
    dataset = NeuralILDataset([_make_pointer_sample()], use_pointer=True)
    item = dataset[0]
    assert item["planet_features"].shape == torch.Size([MAX_PLANETS, 7])


def test_pointer_mode_fleet_features_shape():
    dataset = NeuralILDataset([_make_pointer_sample()], use_pointer=True)
    item = dataset[0]
    assert item["fleet_features"].shape == torch.Size([FLEET_INPUT_DIM])


def test_pointer_mode_planet_mask_shape_and_dtype():
    dataset = NeuralILDataset([_make_pointer_sample(n_real=15)], use_pointer=True)
    item = dataset[0]
    assert item["planet_mask"].shape == torch.Size([MAX_PLANETS])
    # planet_mask should be a bool tensor
    assert item["planet_mask"].dtype == torch.bool


def test_pointer_mode_label_values():
    dataset = NeuralILDataset(
        [_make_pointer_sample(action_type=1, source_idx=2, target_idx=5, amount_bin=3, value_target=0.7)],
        use_pointer=True,
    )
    item = dataset[0]
    assert item["action_type"].item() == 1
    assert item["source_idx"].item() == 2
    assert item["target_idx"].item() == 5
    assert item["amount_bin"].item() == 3
    assert item["value_target"].item() == pytest.approx(0.7)


def test_pointer_mode_mask_counts_real_planets():
    n_real = 7
    dataset = NeuralILDataset([_make_pointer_sample(n_real=n_real)], use_pointer=True)
    item = dataset[0]
    assert item["planet_mask"].sum().item() == n_real


# ---------------------------------------------------------------------------
# Planet-policy mode tests
# ---------------------------------------------------------------------------

def _make_planet_policy_sample(
    n_my_planets=5,
    n_total=10,
    value_target=1.0,
):
    """ILSample with all planet_policy v2 fields populated."""
    planet_features_v2 = np.random.randn(MAX_PLANETS, 10).astype(np.float32)
    fleet_features_v2 = np.random.randn(MAX_PLANETS, 8).astype(np.float32)
    fleet_mask = np.zeros(MAX_PLANETS, dtype=bool)
    fleet_mask[:8] = True
    global_features = np.random.randn(4).astype(np.float32)

    planet_action_types = np.full(MAX_PLANETS, -1, dtype=np.int32)
    planet_target_idxs = np.full(MAX_PLANETS, -1, dtype=np.int32)
    planet_amount_bins = np.full(MAX_PLANETS, -1, dtype=np.int32)
    my_planet_mask = np.zeros(MAX_PLANETS, dtype=bool)

    for i in range(n_my_planets):
        my_planet_mask[i] = True
        planet_action_types[i] = 1  # LAUNCH
        planet_target_idxs[i] = (i + 1) % n_total
        planet_amount_bins[i] = 2

    labels_v2 = PerPlanetLabels(
        planet_action_types=planet_action_types,
        planet_target_idxs=planet_target_idxs,
        planet_amount_bins=planet_amount_bins,
        my_planet_mask=my_planet_mask,
        value_target=value_target,
    )

    return ILSample(
        state_array=np.empty(0, dtype=np.float32),
        labels=ModelLabels(action_type=0, source_idx=-1, target_idx=-1, amount_bin=-1, value_target=value_target),
        planet_features_v2=planet_features_v2,
        fleet_features_v2=fleet_features_v2,
        fleet_mask=fleet_mask,
        global_features=global_features,
        labels_v2=labels_v2,
    )


def test_planet_policy_mode_keys():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    expected = {
        "planet_features", "fleet_features", "fleet_mask", "global_features",
        "planet_mask", "action_types", "target_idxs", "amount_bins", "value_target",
    }
    assert set(item.keys()) == expected


def test_planet_policy_mode_no_state_key():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    assert "state" not in item
    assert "action_type" not in item  # singular — v1 key


def test_planet_policy_planet_features_shape():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    assert item["planet_features"].shape == torch.Size([MAX_PLANETS, 10])


def test_planet_policy_fleet_features_shape():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    assert item["fleet_features"].shape == torch.Size([MAX_PLANETS, 8])


def test_planet_policy_global_features_shape():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    assert item["global_features"].shape == torch.Size([4])


def test_planet_policy_fleet_mask_shape_and_dtype():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    assert item["fleet_mask"].shape == torch.Size([MAX_PLANETS])
    assert item["fleet_mask"].dtype == torch.bool


def test_planet_policy_planet_mask_shape_and_dtype():
    dataset = NeuralILDataset([_make_planet_policy_sample(n_my_planets=5)], use_planet_policy=True)
    item = dataset[0]
    assert item["planet_mask"].shape == torch.Size([MAX_PLANETS])
    assert item["planet_mask"].dtype == torch.bool


def test_planet_policy_planet_mask_count():
    n_my = 5
    dataset = NeuralILDataset([_make_planet_policy_sample(n_my_planets=n_my)], use_planet_policy=True)
    item = dataset[0]
    assert item["planet_mask"].sum().item() == n_my


def test_planet_policy_action_types_shape_and_dtype():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    assert item["action_types"].shape == torch.Size([MAX_PLANETS])
    assert item["action_types"].dtype == torch.long


def test_planet_policy_target_idxs_shape_and_dtype():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    assert item["target_idxs"].shape == torch.Size([MAX_PLANETS])
    assert item["target_idxs"].dtype == torch.long


def test_planet_policy_amount_bins_shape_and_dtype():
    dataset = NeuralILDataset([_make_planet_policy_sample()], use_planet_policy=True)
    item = dataset[0]
    assert item["amount_bins"].shape == torch.Size([MAX_PLANETS])
    assert item["amount_bins"].dtype == torch.long


def test_planet_policy_value_target_dtype():
    dataset = NeuralILDataset([_make_planet_policy_sample(value_target=-1.0)], use_planet_policy=True)
    item = dataset[0]
    assert item["value_target"].dtype == torch.float32
    assert item["value_target"].item() == pytest.approx(-1.0)


def test_planet_policy_padding_slots_are_minus_one():
    """Non-my-planet slots should have action_types=-1 (PADDING / ignore_index)."""
    n_my = 3
    dataset = NeuralILDataset([_make_planet_policy_sample(n_my_planets=n_my)], use_planet_policy=True)
    item = dataset[0]
    # Slots beyond n_my should be -1
    for i in range(n_my, MAX_PLANETS):
        assert item["action_types"][i].item() == -1
        assert item["target_idxs"][i].item() == -1
        assert item["amount_bins"][i].item() == -1


def test_planet_policy_len():
    dataset = NeuralILDataset(
        [_make_planet_policy_sample(), _make_planet_policy_sample()],
        use_planet_policy=True,
    )
    assert len(dataset) == 2
