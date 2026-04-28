"""PyTorch Dataset wrappers for Orbit Wars training samples.

This is the only file in dataset/ that imports torch.
"""

from __future__ import annotations

import bisect
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from dataset.builder import TrainingSample
from dataset.episode import EpisodeReader


class OrbitDataset(Dataset):
    """Eager dataset: receives a pre-built list of TrainingSample."""

    def __init__(
        self,
        samples: list[TrainingSample],
        state_to_tensor: Callable[[Any], torch.Tensor],
        action_to_tensor: Callable[[Any], torch.Tensor],
    ) -> None:
        self._samples = samples
        self._state_to_tensor = state_to_tensor
        self._action_to_tensor = action_to_tensor

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        return {
            "state": self._state_to_tensor(sample.state),
            "action": self._action_to_tensor(sample.action),
            "reward": torch.tensor(
                sample.reward if sample.reward is not None else 0.0,
                dtype=torch.float32,
            ),
            "next_state": (
                self._state_to_tensor(sample.next_state)
                if sample.next_state is not None
                else torch.tensor(0.0)
            ),
            "done": torch.tensor(sample.done, dtype=torch.bool),
        }


class LazyOrbitDataset(Dataset):
    """Lazy dataset — Ciclo C."""

    def __init__(self, catalog, builder, state_to_tensor, action_to_tensor) -> None:
        self._catalog = catalog
        self._builder = builder
        self._state_to_tensor = state_to_tensor
        self._action_to_tensor = action_to_tensor

        if catalog is not None:
            self._episodes = self._catalog.episodes
            self._offsets = [0]
            for meta in self._episodes:
                self._offsets.append(self._offsets[-1] + meta.total_steps)
        else:
            self._episodes = []
            self._offsets = [0]

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> dict:
        ep_idx = bisect.bisect_right(self._offsets, idx) - 1
        local_step = idx - self._offsets[ep_idx]
        meta = self._episodes[ep_idx]
        with EpisodeReader(meta) as reader:
            samples = self._builder.build_episode(reader)
        sample = samples[local_step]
        return {
            "state": self._state_to_tensor(sample.state),
            "action": self._action_to_tensor(sample.action),
            "reward": torch.tensor(
                sample.reward if sample.reward is not None else 0.0,
                dtype=torch.float32,
            ),
            "next_state": (
                self._state_to_tensor(sample.next_state)
                if sample.next_state is not None
                else torch.tensor(0.0)
            ),
            "done": torch.tensor(sample.done, dtype=torch.bool),
        }
