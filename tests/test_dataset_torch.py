"""Torch-dependent tests for dataset/torch_adapter.py."""

import pytest
import numpy as np
from pathlib import Path

torch = pytest.importorskip("torch")

from dataset.catalog import DataCatalog
from dataset.episode import EpisodeReader
from dataset.transforms.state import RawStateTransform
from dataset.transforms.action import RawActionTransform
from dataset.builder import SampleBuilder
from dataset.torch_adapter import OrbitDataset, LazyOrbitDataset

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


def _state_to_tensor(state: dict):
    planets = state["planets"]
    if len(planets) == 0:
        return torch.zeros(1, dtype=torch.float32)
    return torch.tensor(planets.flatten(), dtype=torch.float32)


def _action_to_tensor(action):
    return torch.zeros(1, dtype=torch.float32)


def test_orbit_dataset(meta):
    builder = SampleBuilder(
        state_transform=RawStateTransform(),
        action_transform=RawActionTransform(),
        perspective="winner",
        mode="il_step",
    )
    with EpisodeReader(meta) as reader:
        samples = builder.build_episode(reader)

    dataset = OrbitDataset(samples[:5], _state_to_tensor, _action_to_tensor)
    assert len(dataset) == 5

    item = dataset[0]
    assert set(item.keys()) == {"state", "action", "reward", "next_state", "done"}
    assert item["reward"].dtype == torch.float32
    assert item["done"].dtype == torch.bool


def test_lazy_orbit_dataset_none_catalog_len_zero():
    lazy = LazyOrbitDataset(None, None, None, None)
    assert len(lazy) == 0
