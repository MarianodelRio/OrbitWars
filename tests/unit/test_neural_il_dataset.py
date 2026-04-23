"""Unit tests for NeuralILDataset item shapes, dtypes, and values."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch

from bots.neural.training import NeuralILDataset, ILSample
from bots.neural.types import ModelLabels


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tests
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
