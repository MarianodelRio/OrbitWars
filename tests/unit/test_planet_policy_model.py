"""Unit tests for PlanetPolicyModel forward pass shapes and configuration."""

import pytest

torch = pytest.importorskip("torch")

from bots.neural.planet_policy_model import PlanetPolicyModel, PlanetPolicyConfig, PlanetPolicyOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_cfg(**overrides) -> PlanetPolicyConfig:
    defaults = dict(
        Dp=10, Df=8, Dg=4,
        E=16, F=8, G=32,
        max_planets=10, max_fleets=20,
        n_amount_bins=9, dropout=0.0, n_heads=2, n_layers=2, ffn_hidden=32,
    )
    defaults.update(overrides)
    return PlanetPolicyConfig(**defaults)


def _make_batch(cfg: PlanetPolicyConfig, B, n_real_planets=6, n_real_fleets=8):
    planet_features = torch.randn(B, cfg.max_planets, cfg.Dp)
    fleet_features = torch.randn(B, cfg.max_fleets, cfg.Df)
    global_features = torch.randn(B, cfg.Dg)
    planet_mask = torch.zeros(B, cfg.max_planets, dtype=torch.bool)
    planet_mask[:, :n_real_planets] = True
    fleet_mask = torch.zeros(B, cfg.max_fleets, dtype=torch.bool)
    fleet_mask[:, :n_real_fleets] = True
    relational_tensor = torch.zeros(B, cfg.max_planets, cfg.max_planets, 4)
    return planet_features, fleet_features, fleet_mask, global_features, planet_mask, relational_tensor


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

def test_forward_batch1_output_shapes():
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=1)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.action_type_logits.shape == (1, cfg.max_planets, 3)
    assert out.target_logits.shape == (1, cfg.max_planets, cfg.max_planets)
    assert out.amount_logits.shape == (1, cfg.max_planets, cfg.n_amount_bins)
    assert out.v_outcome.shape == (1, 1)


def test_forward_batch4_output_shapes():
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=4)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.action_type_logits.shape == (4, cfg.max_planets, 3)
    assert out.target_logits.shape == (4, cfg.max_planets, cfg.max_planets)
    assert out.amount_logits.shape == (4, cfg.max_planets, cfg.n_amount_bins)
    assert out.v_outcome.shape == (4, 1)


def test_forward_returns_planet_policy_output():
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=1)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert isinstance(out, PlanetPolicyOutput)


# ---------------------------------------------------------------------------
# Value head tests
# ---------------------------------------------------------------------------

def test_value_in_tanh_range():
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=16)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.v_outcome.min().item() >= -1.0 - 1e-6
    assert out.v_outcome.max().item() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Default config tests
# ---------------------------------------------------------------------------

def test_default_config_shapes():
    cfg = PlanetPolicyConfig()
    model = PlanetPolicyModel(cfg)
    model.eval()
    B = 1
    pf = torch.randn(B, cfg.max_planets, cfg.Dp)
    ff = torch.randn(B, cfg.max_fleets, cfg.Df)
    fm = torch.ones(B, cfg.max_fleets, dtype=torch.bool)
    gf = torch.randn(B, cfg.Dg)
    pm = torch.ones(B, cfg.max_planets, dtype=torch.bool)
    rt = torch.zeros(B, cfg.max_planets, cfg.max_planets, 4)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.action_type_logits.shape == (B, cfg.max_planets, 3)
    assert out.target_logits.shape == (B, cfg.max_planets, cfg.max_planets)
    assert out.amount_logits.shape == (B, cfg.max_planets, cfg.n_amount_bins)
    assert out.v_outcome.shape == (B, 1)


def test_default_config_defaults():
    cfg = PlanetPolicyConfig()
    assert cfg.Dp == 24
    assert cfg.Df == 16
    assert cfg.Dg == 16
    assert cfg.E == 192
    assert cfg.F == 128
    assert cfg.G == 384
    assert cfg.n_heads == 8
    assert cfg.n_layers == 4
    assert cfg.ffn_hidden == 768
    assert cfg.max_planets == 50
    assert cfg.max_fleets == 200
    assert cfg.n_amount_bins == 9


# ---------------------------------------------------------------------------
# Non-default config
# ---------------------------------------------------------------------------

def test_non_default_n_amount_bins():
    cfg = _small_cfg(n_amount_bins=3)
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=1)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.amount_logits.shape == (1, cfg.max_planets, 3)


def test_non_default_max_planets():
    cfg = _small_cfg(max_planets=5)
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=1, n_real_planets=3)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.action_type_logits.shape == (1, 5, 3)
    assert out.target_logits.shape == (1, 5, 5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_planets_masked_does_not_crash():
    """All planets masked (padding only) — should not crash."""
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf = torch.randn(1, cfg.max_planets, cfg.Dp)
    ff = torch.randn(1, cfg.max_fleets, cfg.Df)
    fm = torch.zeros(1, cfg.max_fleets, dtype=torch.bool)
    gf = torch.randn(1, cfg.Dg)
    pm = torch.zeros(1, cfg.max_planets, dtype=torch.bool)
    rt = torch.zeros(1, cfg.max_planets, cfg.max_planets, 4)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.v_outcome.shape == (1, 1)


def test_all_fleets_masked_does_not_crash():
    """No real fleets — pool should produce zero context safely."""
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf = torch.randn(1, cfg.max_planets, cfg.Dp)
    ff = torch.zeros(1, cfg.max_fleets, cfg.Df)
    fm = torch.zeros(1, cfg.max_fleets, dtype=torch.bool)
    gf = torch.randn(1, cfg.Dg)
    pm = torch.ones(1, cfg.max_planets, dtype=torch.bool)
    rt = torch.zeros(1, cfg.max_planets, cfg.max_planets, 4)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.action_type_logits.shape == (1, cfg.max_planets, 3)


def test_single_real_planet_does_not_crash():
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf = torch.randn(1, cfg.max_planets, cfg.Dp)
    ff = torch.randn(1, cfg.max_fleets, cfg.Df)
    fm = torch.ones(1, cfg.max_fleets, dtype=torch.bool)
    gf = torch.randn(1, cfg.Dg)
    pm = torch.zeros(1, cfg.max_planets, dtype=torch.bool)
    pm[0, 0] = True
    rt = torch.zeros(1, cfg.max_planets, cfg.max_planets, 4)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.action_type_logits.shape == (1, cfg.max_planets, 3)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flows_through_model():
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=2)
    out, _ = model(pf, ff, fm, gf, pm, rt)
    loss = out.action_type_logits.mean() + out.v_outcome.mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad


def test_target_logits_are_square_per_planet():
    """target_logits[b, i, j] represents planet i attending to planet j."""
    cfg = _small_cfg(max_planets=8)
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=2, n_real_planets=5)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    # Square attention matrix over the planet axis
    assert out.target_logits.shape == (2, 8, 8)


# ---------------------------------------------------------------------------
# Three value heads
# ---------------------------------------------------------------------------

def test_three_value_heads_shapes():
    cfg = _small_cfg()
    model = PlanetPolicyModel(cfg)
    model.eval()
    pf, ff, fm, gf, pm, rt = _make_batch(cfg=cfg, B=2)
    with torch.no_grad():
        out, _ = model(pf, ff, fm, gf, pm, rt)
    assert out.v_outcome.shape == (2, 1)
    assert out.v_score_diff.shape == (2, 1)
    assert out.v_shaped.shape == (2, 1)
    assert out.v_outcome.abs().max().item() <= 1.0 + 1e-6
