"""Unit tests for RolloutBuffer."""

import numpy as np
import pytest
import torch

from training.rl.rollout_buffer import RolloutBuffer, RolloutStep
from bots.neural.policy_sampler import CanonicalAction, RLMasks


def make_step(reward=0.1, done=False, terminal_reward=0.0, shaped_reward=0.0):
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
        value=0.5,
        reward=reward,
        done=done,
        terminal_reward=terminal_reward,
        shaped_reward=shaped_reward,
        player=0,
        step_count=0,
    )


def test_add_and_is_full():
    buf = RolloutBuffer(capacity=5)
    assert not buf.is_full()
    for i in range(5):
        buf.add(make_step())
    assert buf.is_full()


def test_add_raises_when_full():
    buf = RolloutBuffer(capacity=2)
    buf.add(make_step())
    buf.add(make_step())
    with pytest.raises(ValueError):
        buf.add(make_step())


def test_clear_resets():
    buf = RolloutBuffer(capacity=3)
    for _ in range(3):
        buf.add(make_step())
    buf.clear()
    assert not buf.is_full()
    assert len(buf._steps) == 0


def test_get_batches_keys():
    buf = RolloutBuffer(capacity=6)
    for _ in range(6):
        buf.add(make_step())

    batches = buf.get_batches(2, "cpu")
    assert len(batches) == 3

    required_keys = {
        "planet_features", "fleet_features", "fleet_mask", "planet_mask",
        "global_features", "my_planet_mask", "valid_target_mask",
        "action_types", "target_idxs", "amount_bins",
        "log_prob_old", "value_old", "advantage", "ret",
    }
    for batch in batches:
        for key in required_keys:
            assert key in batch, f"Missing key: {key}"


def test_get_batches_shapes():
    buf = RolloutBuffer(capacity=6)
    for _ in range(6):
        buf.add(make_step())

    batches = buf.get_batches(2, "cpu")
    for batch in batches:
        B = batch["planet_features"].shape[0]
        assert batch["planet_features"].shape == (B, 5, 10)
        assert batch["fleet_features"].shape == (B, 10, 8)
        assert batch["my_planet_mask"].shape == (B, 5)
        assert batch["valid_target_mask"].shape == (B, 5, 5)
        assert batch["action_types"].shape == (B, 5)
        assert batch["log_prob_old"].shape == (B,)
        assert batch["advantage"].shape == (B,)


def test_episode_stats_counts():
    buf = RolloutBuffer(capacity=6)
    # Episode 1: steps 0,1,2 with done=True at step 2
    buf.add(make_step(reward=0.1, done=False, shaped_reward=0.1))
    buf.add(make_step(reward=0.1, done=False, shaped_reward=0.1))
    buf.add(make_step(reward=0.0, done=True, terminal_reward=1.0, shaped_reward=0.0))
    # Episode 2: steps 3,4,5 with done=True at step 5
    buf.add(make_step(reward=0.1, done=False, shaped_reward=0.1))
    buf.add(make_step(reward=0.1, done=False, shaped_reward=0.1))
    buf.add(make_step(reward=0.0, done=True, terminal_reward=-1.0, shaped_reward=0.0))

    stats = buf.episode_stats()
    assert stats["n_episodes"] == 2
    assert stats["total_steps"] == 6
    # ep1: 0.1+0.1+0+1.0 = 1.2, ep2: 0.1+0.1+0+(-1.0) = -0.8, mean = 0.2
    assert abs(stats["mean_ep_reward"] - 0.2) < 1e-5
