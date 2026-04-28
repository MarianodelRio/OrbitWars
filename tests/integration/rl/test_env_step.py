"""Integration tests for OrbitWarsEnv."""

import pytest

from bots.neural.state_builder import StateBuilder
from training.rewards.potential import PotentialReward
from training.envs.orbit_env import OrbitWarsEnv


STRUCTURED_STATE_KEYS = {"planet_features", "fleet_features", "fleet_mask", "planet_mask", "global_features", "context", "relational_tensor"}


def make_env(steps_per_episode=50):
    state_builder = StateBuilder()
    reward_fn = PotentialReward()
    env = OrbitWarsEnv(state_builder, reward_fn, steps_per_episode=steps_per_episode)
    env.set_opponent(lambda obs, config=None: [])
    return env


def test_reset_returns_structured_state():
    env = make_env()
    state, info = env.reset()
    for key in STRUCTURED_STATE_KEYS:
        assert key in state, f"Missing key: {key}"
    assert "player" in info
    assert "step" in info


def test_step_returns_tuple():
    env = make_env()
    state, info = env.reset()
    result = env.step([])
    assert len(result) == 4
    next_state, reward, done, step_info = result
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(step_info, dict)


def test_step_state_has_correct_keys():
    env = make_env()
    state, info = env.reset()
    next_state, reward, done, step_info = env.step([])
    if next_state is not None:
        for key in STRUCTURED_STATE_KEYS:
            assert key in next_state


def test_player_alternates():
    env = make_env()
    _, info1 = env.reset()
    _, info2 = env.reset()
    _, info3 = env.reset()

    assert info1["player"] == 0
    assert info2["player"] == 1
    assert info3["player"] == 0


def test_done_info_contains_terminal_reward():
    env = make_env(steps_per_episode=3)
    state, info = env.reset(player=0)
    done = False
    step_info = {}
    for _ in range(200):
        _, reward, done, step_info = env.step([])
        if done:
            break
    assert done, "Expected done=True within 200 steps for a 3-step episode"
    assert "terminal_reward" in step_info


def test_done_and_reset_cycle():
    env = make_env(steps_per_episode=5)
    state, info = env.reset(player=0)
    done = False
    for _ in range(100):
        _, reward, done, step_info = env.step([])
        if done:
            break
    assert done
    # Should be able to reset after done
    state2, info2 = env.reset(player=0)
    assert state2 is not None
