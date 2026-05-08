"""Unit tests for VecOrbitWarsEnv using a fake env.

FakeEnv and fake_env_factory_* are module-level so they remain
picklable under multiprocessing's 'spawn' start method.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from training.envs.vec_orbit_env import VecOrbitWarsEnv


def _zero_state() -> dict:
    return {
        "planet_features": np.zeros((1, 4), dtype=np.float32),
        "fleet_features": np.zeros((1, 4), dtype=np.float32),
        "fleet_mask": np.zeros((1,), dtype=np.float32),
        "planet_mask": np.zeros((1,), dtype=np.float32),
        "global_features": np.zeros((4,), dtype=np.float32),
        "context": {},
        "relational_tensor": np.zeros((1, 1, 1), dtype=np.float32),
    }


class FakeEnv:
    def __init__(self, steps_until_done: int = 3) -> None:
        self._steps_until_done = steps_until_done
        self._step_count = 0

    def set_opponent(self, fn) -> None:
        self._opponent = fn

    def reset(self, player=None):
        self._step_count = 0
        return _zero_state(), {"player": 0, "step": 0, "episode_count": 1}

    def step(self, actions: list):
        self._step_count += 1
        done = self._step_count >= self._steps_until_done
        info = {
            "step": self._step_count,
            "shaped_reward": 0.0,
            "terminal_reward": 0.0,
        }
        return _zero_state(), 0.0, done, info


def fake_env_factory_long() -> FakeEnv:
    return FakeEnv(steps_until_done=10)


def fake_env_factory_short() -> FakeEnv:
    return FakeEnv(steps_until_done=1)


REQUIRED_KEYS = {
    "planet_features",
    "fleet_features",
    "fleet_mask",
    "planet_mask",
    "global_features",
    "context",
    "relational_tensor",
}


def test_vec_env_reset_returns_n_states():
    vec = VecOrbitWarsEnv(
        n_envs=3,
        env_factory=fake_env_factory_long,
        opponent_fns=[None, None, None],
    )
    try:
        states = vec.reset()
        assert len(states) == 3
        for s in states:
            assert isinstance(s, dict)
            assert REQUIRED_KEYS.issubset(s.keys())
    finally:
        vec.close()


def test_vec_env_step_advances_all():
    vec = VecOrbitWarsEnv(
        n_envs=3,
        env_factory=fake_env_factory_long,
        opponent_fns=[None, None, None],
    )
    try:
        vec.reset()
        results = vec.step([[], [], []])
        assert len(results) == 3
        for state, reward, done, info in results:
            assert isinstance(state, dict)
            assert REQUIRED_KEYS.issubset(state.keys())
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
    finally:
        vec.close()


def test_vec_env_auto_reset_on_done():
    vec = VecOrbitWarsEnv(
        n_envs=3,
        env_factory=fake_env_factory_short,
        opponent_fns=[None, None, None],
    )
    try:
        vec.reset()
        results = vec.step([[], [], []])
        for state, reward, done, info in results:
            assert done is True
            assert "reset_state" in info
            assert REQUIRED_KEYS.issubset(info["reset_state"].keys())
    finally:
        vec.close()


def test_vec_env_close_terminates_workers():
    vec = VecOrbitWarsEnv(
        n_envs=3,
        env_factory=fake_env_factory_long,
        opponent_fns=[None, None, None],
    )
    procs = list(vec._procs)
    try:
        vec.reset()
        vec.close()
        deadline = time.time() + 3.0
        while time.time() < deadline and any(p.is_alive() for p in procs):
            time.sleep(0.05)
        for p in procs:
            assert not p.is_alive()
    finally:
        vec.close()
