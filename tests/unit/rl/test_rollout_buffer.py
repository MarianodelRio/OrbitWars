"""Unit tests for RolloutBuffer."""

import math

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
        "relational_tensor": np.zeros((max_planets, max_planets, 4), dtype=np.float32),
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
        "global_features", "relational_tensor", "my_planet_mask", "valid_target_mask",
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


def test_episode_stats_win_loss_draw_counts():
    buf = RolloutBuffer(capacity=3)
    buf.add(make_step(done=True, terminal_reward=10.0))   # win
    buf.add(make_step(done=True, terminal_reward=-10.0))  # loss
    buf.add(make_step(done=True, terminal_reward=0.0))    # draw
    stats = buf.episode_stats()
    assert stats["n_wins"] == 1
    assert stats["n_losses"] == 1
    assert stats["n_draws"] == 1
    assert stats["win_rate"] == pytest.approx(1 / 3)


def test_episode_stats_mean_ep_length():
    buf = RolloutBuffer(capacity=9)
    # Episode 1: 3 non-done + 1 done = 4 steps
    for _ in range(3):
        buf.add(make_step(done=False))
    buf.add(make_step(done=True))
    # Episode 2: 4 non-done + 1 done = 5 steps
    for _ in range(4):
        buf.add(make_step(done=False))
    buf.add(make_step(done=True))
    stats = buf.episode_stats()
    assert stats["mean_ep_length"] == pytest.approx(4.5)


def test_episode_stats_adv_mean_finite():
    buf = RolloutBuffer(capacity=4)
    for _ in range(3):
        buf.add(make_step(done=False))
    buf.add(make_step(done=True))
    buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    stats = buf.episode_stats()
    assert math.isfinite(stats["adv_mean"])
    assert math.isfinite(stats["adv_std"])


def test_get_batches_with_lstm_state():
    max_planets = 5
    G = 4

    def make_lstm_step():
        state = {
            "planet_features": np.zeros((max_planets, 10), dtype=np.float32),
            "fleet_features": np.zeros((10, 8), dtype=np.float32),
            "fleet_mask": np.zeros(10, dtype=bool),
            "planet_mask": np.zeros(max_planets, dtype=bool),
            "global_features": np.zeros(4, dtype=np.float32),
            "relational_tensor": np.zeros((max_planets, max_planets, 4), dtype=np.float32),
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
            reward=0.1,
            done=False,
            terminal_reward=0.0,
            shaped_reward=0.0,
            player=0,
            step_count=0,
            h_n=torch.zeros(1, 1, G),
            c_n=torch.zeros(1, 1, G),
        )

    buf = RolloutBuffer(capacity=4)
    for _ in range(4):
        buf.add(make_lstm_step())

    batches = buf.get_batches(4, "cpu")
    assert len(batches) == 1
    batch = batches[0]
    assert "h_n" in batch
    assert "c_n" in batch
    B = batch["planet_features"].shape[0]
    assert batch["h_n"].shape == (B, 1, G)
    assert batch["c_n"].shape == (B, 1, G)
