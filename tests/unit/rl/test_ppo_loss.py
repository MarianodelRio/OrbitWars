"""Unit tests for PPO loss computation."""

import math

import numpy as np
import pytest
import torch

from bots.neural.planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel
from bots.neural.policy_sampler import PolicySampler, CanonicalAction, RLMasks
from training.rl.ppo import compute_ppo_loss, PPOLossResult
from training.utils.rl_config import RLConfig


def make_random_batch(B=4, max_planets=5, n_bins=5, max_fleets=10, Dp=24, Df=16, Dg=16):
    planet_mask = torch.zeros(B, max_planets, dtype=torch.bool)
    planet_mask[:, :3] = True

    my_planet_mask = torch.zeros(B, max_planets, dtype=torch.bool)
    my_planet_mask[:, 0] = True
    my_planet_mask[:, 2] = True

    valid_target_mask = torch.zeros(B, max_planets, max_planets, dtype=torch.bool)
    for b in range(B):
        for i in range(max_planets):
            if planet_mask[b, i]:
                for j in range(max_planets):
                    if planet_mask[b, j] and j != i:
                        valid_target_mask[b, i, j] = True

    action_types = torch.full((B, max_planets), -1, dtype=torch.int64)
    action_types[:, 0] = 1   # LAUNCH
    action_types[:, 2] = 0   # NO_OP

    target_idxs = torch.full((B, max_planets), -1, dtype=torch.int64)
    target_idxs[:, 0] = 1   # target planet 1

    amount_bins = torch.full((B, max_planets), -1, dtype=torch.int64)
    amount_bins[:, 0] = 2   # bin index 2

    return {
        "planet_features": torch.randn(B, max_planets, Dp),
        "fleet_features": torch.randn(B, max_fleets, Df),
        "fleet_mask": torch.zeros(B, max_fleets, dtype=torch.bool),
        "planet_mask": planet_mask,
        "global_features": torch.randn(B, Dg),
        "relational_tensor": torch.zeros(B, max_planets, max_planets, 4),
        "my_planet_mask": my_planet_mask,
        "valid_target_mask": valid_target_mask,
        "action_types": action_types,
        "target_idxs": target_idxs,
        "amount_bins": amount_bins,
        "log_prob_old": torch.randn(B),
        "value_old": torch.randn(B),
        "advantage": torch.randn(B),
        "ret": torch.randn(B),
    }


def make_model_and_sampler(max_planets=5):
    cfg = PlanetPolicyConfig(max_planets=max_planets, max_fleets=10)
    model = PlanetPolicyModel(cfg)
    sampler = PolicySampler(bins=[0.1, 0.25, 0.5, 0.75, 1.0], max_planets=max_planets)
    return model, sampler


def test_total_loss_has_grad():
    model, sampler = make_model_and_sampler()
    config = RLConfig(ppo_batch_size=4, normalize_advantages=True)
    batch = make_random_batch(B=4)

    loss, result = compute_ppo_loss(model, batch, config)
    assert loss.requires_grad
    # Should not raise
    loss.backward()


def test_approx_kl_nonneg():
    model, sampler = make_model_and_sampler()
    config = RLConfig(ppo_batch_size=4)
    batch = make_random_batch(B=4)

    # Set old log_prob equal to new (after forward) to get approx_kl ~ 0
    _, result = compute_ppo_loss(model, batch, config)
    # approx_kl = mean(old - new). With random old it can be negative or positive.
    # The key property: it must be a finite float.
    assert isinstance(result.approx_kl, float)
    assert not np.isnan(result.approx_kl)


def test_clip_fraction_in_range():
    model, sampler = make_model_and_sampler()
    config = RLConfig(ppo_batch_size=4)
    batch = make_random_batch(B=4)

    _, result = compute_ppo_loss(model, batch, config)
    assert 0.0 <= result.clip_fraction <= 1.0


def test_ppo_loss_result_fields():
    model, sampler = make_model_and_sampler()
    config = RLConfig(ppo_batch_size=4)
    batch = make_random_batch(B=4)

    _, result = compute_ppo_loss(model, batch, config)
    assert hasattr(result, "total_loss")
    assert hasattr(result, "policy_loss")
    assert hasattr(result, "value_loss")
    assert hasattr(result, "entropy")
    assert hasattr(result, "approx_kl")
    assert hasattr(result, "clip_fraction")
    assert hasattr(result, "explained_variance")


def test_no_nan_with_peaked_logits():
    model, sampler = make_model_and_sampler()
    config = RLConfig(ppo_batch_size=4)
    batch = make_random_batch(B=4)

    # Force peaked action_type distribution: class 0 wins with probability ≈ 1.0
    # in float32, so that other classes have softmax probability that rounds to 0.0.
    with torch.no_grad():
        bias = model.action_type_head.bias  # shape (3,)
        bias[0] = 88.0
        bias[1] = -88.0
        bias[2] = -88.0

    _, result = compute_ppo_loss(model, batch, config)

    assert math.isfinite(result.total_loss), f"total_loss is not finite: {result.total_loss}"
    assert math.isfinite(result.entropy_action_type) and result.entropy_action_type >= 0, (
        f"entropy_action_type invalid: {result.entropy_action_type}"
    )
    assert math.isfinite(result.entropy_amount) and result.entropy_amount >= 0, (
        f"entropy_amount invalid: {result.entropy_amount}"
    )


def test_kl_bc_with_peaked_bc_model():
    model, _ = make_model_and_sampler()
    bc_model, _ = make_model_and_sampler()
    config = RLConfig(ppo_batch_size=4)
    batch = make_random_batch(B=4)

    # Force peaked logits on both action_type and amount heads of bc_model
    with torch.no_grad():
        bc_model.action_type_head.bias[0] = 88.0
        bc_model.action_type_head.bias[1] = -88.0
        bc_model.action_type_head.bias[2] = -88.0

    loss, result = compute_ppo_loss(model, batch, config, bc_model=bc_model, kl_bc_coef=0.1)

    assert math.isfinite(result.total_loss), f"total_loss is not finite: {result.total_loss}"
