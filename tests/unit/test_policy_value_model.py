"""Unit tests for PolicyValueModel and PointerNetworkModel forward pass shapes and configuration."""

import torch
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
from bots.neural.model import PolicyValueModel, PolicyValueConfig
from bots.neural.pointer_model import PointerNetworkModel, PointerNetworkConfig


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


# ---------------------------------------------------------------------------
# PointerNetworkModel
# ---------------------------------------------------------------------------

def _make_pointer_batch(B=2, max_planets=50, n_real=10, fleet_input_dim=700):
    planet_features = torch.randn(B, max_planets, 7)
    fleet_features = torch.randn(B, fleet_input_dim)
    planet_mask = torch.zeros(B, max_planets, dtype=torch.bool)
    planet_mask[:, :n_real] = True
    return planet_features, fleet_features, planet_mask


def test_pointer_forward_output_shapes_batch1():
    cfg = PointerNetworkConfig()
    model = PointerNetworkModel(cfg)
    model.eval()
    pf, ff, pm = _make_pointer_batch(B=1)
    with torch.no_grad():
        out = model(pf, ff, pm)
    assert out.action_type_logits.shape == (1, 2)
    assert out.source_logits.shape == (1, 50)
    assert out.target_logits.shape == (1, 50)
    assert out.amount_logits.shape == (1, 5)
    assert out.value.shape == (1, 1)


def test_pointer_forward_output_shapes_batch4():
    cfg = PointerNetworkConfig()
    model = PointerNetworkModel(cfg)
    model.eval()
    pf, ff, pm = _make_pointer_batch(B=4)
    with torch.no_grad():
        out = model(pf, ff, pm)
    assert out.action_type_logits.shape[0] == 4
    assert out.source_logits.shape[0] == 4
    assert out.value.shape[0] == 4


def test_pointer_value_in_tanh_range():
    cfg = PointerNetworkConfig()
    model = PointerNetworkModel(cfg)
    model.eval()
    pf, ff, pm = _make_pointer_batch(B=8)
    with torch.no_grad():
        out = model(pf, ff, pm)
    assert out.value.min().item() >= -1.0 - 1e-6
    assert out.value.max().item() <= 1.0 + 1e-6


def test_pointer_padding_positions_are_neginf():
    """Padding positions (planet_mask=False) should have -inf logits."""
    cfg = PointerNetworkConfig(max_planets=10)
    model = PointerNetworkModel(cfg)
    model.eval()
    B = 1
    planet_features = torch.randn(B, 10, 7)
    fleet_features = torch.randn(B, 700)
    planet_mask = torch.zeros(B, 10, dtype=torch.bool)
    planet_mask[:, :5] = True   # only first 5 real
    with torch.no_grad():
        out = model(planet_features, fleet_features, planet_mask)
    # Slots 5-9 should be -inf in source_logits and target_logits
    assert torch.all(out.source_logits[:, 5:] == float("-inf"))
    assert torch.all(out.target_logits[:, 5:] == float("-inf"))


def test_pointer_non_default_n_amount_bins():
    cfg = PointerNetworkConfig(n_amount_bins=3)
    model = PointerNetworkModel(cfg)
    model.eval()
    pf, ff, pm = _make_pointer_batch(B=1)
    with torch.no_grad():
        out = model(pf, ff, pm)
    assert out.amount_logits.shape == (1, 3)


def test_pointer_single_planet_does_not_crash():
    """Edge case: only one real planet — mask prevents trivial failure."""
    cfg = PointerNetworkConfig(max_planets=5)
    model = PointerNetworkModel(cfg)
    model.eval()
    pf = torch.randn(1, 5, 7)
    ff = torch.randn(1, 700)
    pm = torch.zeros(1, 5, dtype=torch.bool)
    pm[:, 0] = True   # only planet 0 is real
    with torch.no_grad():
        out = model(pf, ff, pm)
    assert out.action_type_logits.shape == (1, 2)
    assert out.source_logits[0, 0].item() != float("-inf")
    assert torch.all(out.source_logits[:, 1:] == float("-inf"))
