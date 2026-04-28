"""Tests for the dataset/ module (Ciclo A + B) — no torch dependency."""

import numpy as np
import pytest
from pathlib import Path

from dataset.catalog import DataCatalog, EpisodeMeta
from dataset.episode import EpisodeReader, StepRecord
from dataset.transforms.state import RawStateTransform
from dataset.transforms.action import RawActionTransform
from dataset.transforms.reward import BinaryOutcomeReward
from dataset.transforms.filters import HasActionFilter, EarlyGameFilter, CompositeFilter
from dataset.builder import SampleBuilder, TrainingSample

H5_PATH = Path("/home/mariano/Desktop/OrbitWars/data/matches/scoring.bot_vs_heuristic.baseline/20260421_172107_match_0001.h5")
_H5_AVAILABLE = H5_PATH.exists()


@pytest.fixture(scope="module")
def catalog():
    if not _H5_AVAILABLE:
        pytest.skip("H5 test data not available at H5_PATH")
    return DataCatalog.scan(roots=[H5_PATH.parent])


@pytest.fixture(scope="module")
def meta(catalog):
    return catalog.episodes[0]


def _dummy_step(turn: int, n_actions_p0: int = 0) -> StepRecord:
    return StepRecord(
        turn=turn,
        planets=np.empty((0, 7), dtype=np.float32),
        fleets=np.empty((0, 7), dtype=np.float32),
        actions_p0=np.ones((n_actions_p0, 3), dtype=np.float32) if n_actions_p0 > 0 else np.empty((0, 3), dtype=np.float32),
        actions_p1=np.empty((0, 3), dtype=np.float32),
        comet_planet_ids=np.empty((0,), dtype=np.int32),
        is_terminal=False,
    )


# --- DataCatalog ---

def test_catalog_scan(catalog):
    assert len(catalog) == 1
    ep = catalog.episodes[0]
    assert ep.bot0 == "scoring.bot"
    assert ep.winner == 0
    assert ep.total_steps == 347


def test_catalog_filter_winner_only(catalog):
    f1 = catalog.filter(bot="scoring.bot", winner_only=True)
    assert len(f1) == 1

    f2 = catalog.filter(bot="heuristic.baseline", winner_only=True)
    assert len(f2) == 0


# --- EpisodeReader ---

def test_episode_reader_step_count(meta):
    with EpisodeReader(meta) as reader:
        assert len(list(reader.steps())) == 347


def test_episode_reader_no_padding(meta):
    with EpisodeReader(meta) as reader:
        step = reader.step(1)
    assert step.planets.shape == (24, 7)
    assert step.fleets.shape[1] == 7


def test_episode_reader_terminal(meta):
    with EpisodeReader(meta) as reader:
        assert reader.step(346).is_terminal is True
        with pytest.raises(IndexError):
            reader.step(347)


def test_step_record_turn_zero(meta):
    with EpisodeReader(meta) as reader:
        step = reader.step(0)
    assert step.planets.shape == (0, 7)
    assert step.turn == 0


# --- Filters ---

def test_has_action_filter(meta):
    f = HasActionFilter()
    with EpisodeReader(meta) as reader:
        step0 = reader.step(0)
    assert f(step0, 0) is False


def test_early_game_filter():
    f = EarlyGameFilter(max_turn=10)
    assert f(_dummy_step(10), 0) is True
    assert f(_dummy_step(11), 0) is False


def test_composite_filter():
    cf = CompositeFilter(HasActionFilter(), EarlyGameFilter(max_turn=10))
    # no actions, turn 5 → HasActionFilter fails → False
    assert cf(_dummy_step(5, n_actions_p0=0), 0) is False
    # has actions, turn 5 → both pass → True
    assert cf(_dummy_step(5, n_actions_p0=1), 0) is True
    # has actions, turn 11 → EarlyGameFilter fails → False
    assert cf(_dummy_step(11, n_actions_p0=1), 0) is False


# --- SampleBuilder ---

def test_sample_builder_il_step(meta):
    builder = SampleBuilder(
        state_transform=RawStateTransform(),
        action_transform=RawActionTransform(),
        perspective="winner",
        mode="il_step",
    )
    with EpisodeReader(meta) as reader:
        samples = builder.build_episode(reader)

    assert len(samples) == 347
    for s in samples:
        assert s.reward is None
        assert s.next_state is None


def test_sample_builder_rl_transition(meta):
    builder = SampleBuilder(
        state_transform=RawStateTransform(),
        action_transform=RawActionTransform(),
        reward_transform=BinaryOutcomeReward(),
        perspective="winner",
        mode="rl_transition",
    )
    with EpisodeReader(meta) as reader:
        samples = builder.build_episode(reader)

    assert len(samples) == 347

    last = samples[-1]
    assert last.done is True
    assert last.next_state is None
    assert last.reward == 1.0  # player 0 won

    first = samples[0]
    assert first.done is False
    assert first.reward == 0.0
    assert first.next_state is not None


def test_sample_builder_rl_transition_requires_reward(meta):
    with pytest.raises(ValueError):
        SampleBuilder(
            state_transform=RawStateTransform(),
            action_transform=RawActionTransform(),
            mode="rl_transition",
        )

