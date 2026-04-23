"""Unit tests for PolicyValueModel forward pass shapes and configuration."""

import torch
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
from bots.neural.model import PolicyValueModel, PolicyValueConfig


def test_forward_batch1_output_shapes():
    config = PolicyValueConfig(input_dim=32, hidden_dims=[64, 32])
    model = PolicyValueModel(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 32))
    assert output.action_type_logits.shape == (1, 2)
    assert output.source_logits.shape == (1, 50)
    assert output.target_logits.shape == (1, 50)
    assert output.amount_logits.shape == (1, 5)
    assert output.value.shape == (1, 1)


def test_forward_batch4_output_shapes():
    config = PolicyValueConfig(input_dim=32, hidden_dims=[64, 32])
    model = PolicyValueModel(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(4, 32))
    assert output.action_type_logits.shape[0] == 4
    assert output.source_logits.shape[0] == 4
    assert output.target_logits.shape[0] == 4
    assert output.amount_logits.shape[0] == 4
    assert output.value.shape[0] == 4


def test_value_in_tanh_range():
    config = PolicyValueConfig(input_dim=32, hidden_dims=[64, 32])
    model = PolicyValueModel(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(16, 32))
    assert output.value.min().item() >= -1.0 - 1e-6
    assert output.value.max().item() <= 1.0 + 1e-6


def test_non_default_max_planets():
    config = PolicyValueConfig(input_dim=16, hidden_dims=[32], max_planets=10)
    model = PolicyValueModel(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 16))
    assert output.source_logits.shape == (1, 10)
    assert output.target_logits.shape == (1, 10)


def test_non_default_n_amount_bins():
    config = PolicyValueConfig(input_dim=16, hidden_dims=[32], n_amount_bins=3)
    model = PolicyValueModel(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 16))
    assert output.amount_logits.shape == (1, 3)


def test_encoder_layers_respect_hidden_dims():
    config = PolicyValueConfig(input_dim=8, hidden_dims=[64, 32])
    model = PolicyValueModel(config)
    assert isinstance(model.encoder, nn.Sequential)
    linear_layers = [m for m in model.encoder if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 2
    assert isinstance(model.value_head, nn.Linear)
