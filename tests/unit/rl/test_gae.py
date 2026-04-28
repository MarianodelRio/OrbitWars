"""Unit tests for GAE computation."""

import pytest

from training.rl.gae import compute_gae
from training.rl.rollout_buffer import RolloutStep
from bots.neural.policy_sampler import CanonicalAction, RLMasks
import numpy as np
import torch


def make_step(reward, value, done=False):
    max_planets = 5
    state = {
        "planet_features": np.zeros((max_planets, 10), dtype=np.float32),
        "fleet_features": np.zeros((10, 8), dtype=np.float32),
        "fleet_mask": np.zeros(10, dtype=bool),
        "planet_mask": np.zeros(max_planets, dtype=bool),
        "global_features": np.zeros(4, dtype=np.float32),
    }
    rl_masks = RLMasks(
        my_planet_mask=torch.zeros(max_planets, dtype=torch.bool),
        valid_target_mask=torch.zeros(max_planets, max_planets, dtype=torch.bool),
        planet_mask=torch.zeros(max_planets, dtype=torch.bool),
    )
    canonical = CanonicalAction(
        action_types=np.full(max_planets, -1, dtype=np.int8),
        target_idxs=np.full(max_planets, -1, dtype=np.int16),
        amount_bins=np.full(max_planets, -1, dtype=np.int8),
    )
    return RolloutStep(
        state=state,
        rl_masks=rl_masks,
        canonical=canonical,
        log_prob_old=0.0,
        value=value,
        reward=reward,
        done=done,
        terminal_reward=0.0,
        shaped_reward=0.0,
        player=0,
        step_count=0,
    )


def test_gae_advantage_plus_value_equals_ret():
    steps = [make_step(1.0, 0.5), make_step(0.5, 0.3), make_step(0.2, 0.1)]
    compute_gae(steps, last_value=0.0, gamma=0.99, gae_lambda=0.95)
    for s in steps:
        assert abs(s.advantage + s.value - s.ret) < 1e-5


def test_gae_known_values():
    # Single step, last_value=0, done=False
    # delta = r + gamma*0 - v = 1.0 + 0 - 0.5 = 0.5
    # gae = delta + 0 = 0.5
    # ret = 0.5 + 0.5 = 1.0
    steps = [make_step(1.0, 0.5, done=False)]
    compute_gae(steps, last_value=0.0, gamma=0.99, gae_lambda=0.95)
    assert abs(steps[0].advantage - 0.5) < 1e-5
    assert abs(steps[0].ret - 1.0) < 1e-5


def test_gae_with_done_mid_sequence():
    # Step 0: reward=1.0, value=0.5, done=True -> terminal, next_value should be 0
    # Step 1: reward=0.5, value=0.3, done=False
    step0 = make_step(1.0, 0.5, done=True)
    step1 = make_step(0.5, 0.3, done=False)
    steps = [step0, step1]
    compute_gae(steps, last_value=0.2, gamma=0.99, gae_lambda=0.95)

    # For step 1 (t=1, last step):
    # delta1 = 0.5 + 0.99 * 0.2 * 1.0 - 0.3 = 0.5 + 0.198 - 0.3 = 0.398
    # gae1 = 0.398
    assert abs(steps[1].advantage - 0.398) < 1e-4

    # For step 0 (t=0, done=True):
    # next_non_terminal = 0.0
    # delta0 = 1.0 + 0.99 * steps[1].value * 0.0 - 0.5 = 0.5
    # gae0 = 0.5 + 0.99 * 0.95 * 0.0 * gae1 = 0.5
    assert abs(steps[0].advantage - 0.5) < 1e-5


def test_gae_three_steps_manual():
    gamma = 0.99
    lam = 0.95
    steps = [
        make_step(reward=1.0, value=0.5),
        make_step(reward=0.5, value=0.3),
        make_step(reward=0.2, value=0.1),
    ]
    last_value = 0.0
    compute_gae(steps, last_value, gamma, lam)

    # Manual computation:
    # t=2: delta2 = 0.2 + 0.99*0.0 - 0.1 = 0.1; gae2 = 0.1
    # t=1: delta1 = 0.5 + 0.99*0.1 - 0.3 = 0.299; gae1 = 0.299 + 0.99*0.95*0.1 = 0.299 + 0.09405 = 0.39305
    # t=0: delta0 = 1.0 + 0.99*0.3 - 0.5 = 0.797; gae0 = 0.797 + 0.99*0.95*0.39305 = 0.797 + 0.36975...
    delta2 = 0.2 + gamma * 0.0 - 0.1
    gae2 = delta2
    delta1 = 0.5 + gamma * 0.1 - 0.3
    gae1 = delta1 + gamma * lam * gae2
    delta0 = 1.0 + gamma * 0.3 - 0.5
    gae0 = delta0 + gamma * lam * gae1

    assert abs(steps[2].advantage - gae2) < 1e-5
    assert abs(steps[1].advantage - gae1) < 1e-5
    assert abs(steps[0].advantage - gae0) < 1e-5
