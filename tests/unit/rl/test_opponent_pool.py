"""Unit tests for OpponentPool — never calls get_agent() to avoid kaggle_environments."""

import pathlib

import pytest

from training.rl.opponent_pool import OpponentPool, PoolEntry


def test_empty_pool_sample_returns_noop():
    pool = OpponentPool()
    fn = pool.sample()
    assert callable(fn)
    assert fn(None) == []


def test_add_heuristic_increases_size():
    pool = OpponentPool()
    pool.add_heuristic("baseline", "bots.heuristic.baseline:agent_fn")
    assert pool.size() == 1


def test_add_snapshot_increases_size():
    pool = OpponentPool()
    pool.add_snapshot(pathlib.Path("fake.pt"), iteration=1)
    assert pool.size() == 1


def test_snapshot_eviction_when_max_exceeded():
    pool = OpponentPool(max_snapshots=2)
    pool.add_snapshot(pathlib.Path("snap_0.pt"), iteration=0)
    pool.add_snapshot(pathlib.Path("snap_1.pt"), iteration=1)
    pool.add_snapshot(pathlib.Path("snap_2.pt"), iteration=2)
    # max_snapshots=2: snap_0 should be evicted
    assert pool.size() == 2
    snapshot_names = [e.name for e in pool._snapshot_entries]
    assert "snapshot:0" not in snapshot_names
    assert "snapshot:1" in snapshot_names
    assert "snapshot:2" in snapshot_names


def test_heuristics_survive_snapshot_eviction():
    pool = OpponentPool(max_snapshots=1)
    pool.add_heuristic("baseline", "bots.heuristic.baseline:agent_fn")
    pool.add_snapshot(pathlib.Path("snap_0.pt"), iteration=0)
    pool.add_snapshot(pathlib.Path("snap_1.pt"), iteration=1)
    # snap_0 evicted, heuristic survives
    heuristic_names = [e.name for e in pool._entries if e.name == "baseline"]
    assert len(heuristic_names) == 1


def test_sample_returns_callable():
    pool = OpponentPool()
    # Inject a dummy entry directly to avoid calling load_agent
    dummy_entry = PoolEntry(name="dummy", loader=lambda: (lambda obs, config=None: []))
    pool._entries.append(dummy_entry)
    fn = pool.sample()
    assert callable(fn)


def test_size_counts_all_entry_types():
    pool = OpponentPool(max_snapshots=5)
    pool.add_heuristic("h1", "bots.heuristic.baseline:agent_fn")
    pool.add_snapshot(pathlib.Path("snap_0.pt"), iteration=0)
    pool.add_snapshot(pathlib.Path("snap_1.pt"), iteration=1)
    assert pool.size() == 3
