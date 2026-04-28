"""Unit tests for PolicySampler."""

import math

import numpy as np
import pytest
import torch

from bots.neural.planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel, PlanetPolicyOutput
from bots.neural.policy_sampler import PolicySampler, CanonicalAction, RLMasks
from bots.neural.types import ActionContext
from bots.neural.action_codec import ActionCodec


def make_context(n=3, player=0):
    planet_ids = np.arange(n, dtype=np.int32)
    planet_positions = np.array([[float(i * 10), float(i * 10)] for i in range(n)], dtype=np.float32)
    my_planet_mask = np.array([True, False, True][:n], dtype=bool)
    return ActionContext(
        planet_ids=planet_ids,
        planet_positions=planet_positions,
        my_planet_mask=my_planet_mask,
        n_planets=n,
    )


def make_output(max_planets=10):
    cfg = PlanetPolicyConfig(max_planets=max_planets)
    model = PlanetPolicyModel(cfg)
    pf = torch.zeros(1, max_planets, cfg.Dp)
    ff = torch.zeros(1, cfg.max_fleets, cfg.Df)
    fm = torch.zeros(1, cfg.max_fleets, dtype=torch.bool)
    gf = torch.zeros(1, cfg.Dg)
    pm = torch.zeros(1, max_planets, dtype=torch.bool)
    pm[0, :3] = True
    with torch.no_grad():
        out_batched, _ = model(pf, ff, fm, gf, pm)
    output = PlanetPolicyOutput(
        action_type_logits=out_batched.action_type_logits.squeeze(0),
        target_logits=out_batched.target_logits.squeeze(0),
        amount_logits=out_batched.amount_logits.squeeze(0),
        v_outcome=out_batched.v_outcome.squeeze(0),
        v_score_diff=out_batched.v_score_diff.squeeze(0),
        v_shaped=out_batched.v_shaped.squeeze(0),
    )
    return output


def test_build_masks_shapes():
    max_planets = 10
    sampler = PolicySampler(bins=ActionCodec().BINS, max_planets=max_planets)
    ctx = make_context(3)
    masks = sampler.build_masks(ctx, device="cpu")

    assert masks.my_planet_mask.shape == (max_planets,)
    assert masks.valid_target_mask.shape == (max_planets, max_planets)
    assert masks.planet_mask.shape == (max_planets,)


def test_build_masks_values():
    max_planets = 10
    sampler = PolicySampler(bins=[0.1, 0.25, 0.5, 0.75, 1.0], max_planets=max_planets)
    ctx = make_context(3)
    masks = sampler.build_masks(ctx, device="cpu")

    # First 3 slots should be active planets
    assert masks.planet_mask[:3].all()
    assert not masks.planet_mask[3:].any()

    # my_planet_mask: positions 0 and 2 are mine
    assert masks.my_planet_mask[0].item()
    assert not masks.my_planet_mask[1].item()
    assert masks.my_planet_mask[2].item()

    # valid_target_mask: self should be excluded
    for i in range(max_planets):
        assert not masks.valid_target_mask[i, i].item()


def test_sample_deterministic_reproducible():
    max_planets = 10
    sampler = PolicySampler(bins=[0.1, 0.25, 0.5, 0.75, 1.0], max_planets=max_planets)
    ctx = make_context(3)
    output = make_output(max_planets)
    masks = sampler.build_masks(ctx, device="cpu")
    planet_features = np.zeros((max_planets, 10), dtype=np.float32)
    planet_features[:3, 5] = 0.5  # ships/200

    result1 = sampler.sample(output, masks, ctx, planet_features, deterministic=True)
    result2 = sampler.sample(output, masks, ctx, planet_features, deterministic=True)

    np.testing.assert_array_equal(result1.canonical.action_types, result2.canonical.action_types)
    np.testing.assert_array_equal(result1.canonical.target_idxs, result2.canonical.target_idxs)
    np.testing.assert_array_equal(result1.canonical.amount_bins, result2.canonical.amount_bins)


def test_game_actions_no_small_ships():
    max_planets = 10
    sampler = PolicySampler(bins=[0.1, 0.25, 0.5, 0.75, 1.0], max_planets=max_planets)
    ctx = make_context(3)
    output = make_output(max_planets)
    masks = sampler.build_masks(ctx, device="cpu")
    planet_features = np.zeros((max_planets, 10), dtype=np.float32)
    planet_features[:3, 5] = 0.005  # very few ships -> n_ships < 1 for small bins

    result = sampler.sample(output, masks, ctx, planet_features, deterministic=True)
    for action in result.game_actions:
        assert action[2] >= 1.0, f"action has n_ships={action[2]} < 1.0"


def test_compute_log_prob_matches_sample():
    max_planets = 10
    sampler = PolicySampler(bins=[0.1, 0.25, 0.5, 0.75, 1.0], max_planets=max_planets)
    ctx = make_context(3)
    output = make_output(max_planets)
    masks = sampler.build_masks(ctx, device="cpu")
    planet_features = np.zeros((max_planets, 10), dtype=np.float32)
    planet_features[:3, 5] = 0.5

    result = sampler.sample(output, masks, ctx, planet_features, deterministic=True)
    recomputed_lp = sampler.compute_log_prob(output, masks, result.canonical)

    assert abs(result.log_prob.item() - recomputed_lp.item()) < 1e-4


def test_entropy_nonnegative():
    max_planets = 10
    sampler = PolicySampler(bins=[0.1, 0.25, 0.5, 0.75, 1.0], max_planets=max_planets)
    ctx = make_context(3)
    output = make_output(max_planets)
    masks = sampler.build_masks(ctx, device="cpu")

    entropy = sampler.compute_entropy(output, masks)
    assert entropy.item() >= 0.0
