"""NeuralILDataset and build_il_dataset — lazy imitation learning dataset pipeline.

NeuralILDataset stores a lightweight index of (episode, step, player) tuples and
builds each sample on demand in __getitem__.  An LRU cache of open EpisodeReaders
(cache=True) amortises HDF5 I/O across all steps of the same episode, keeping
memory bounded regardless of dataset size.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.catalog import DataCatalog, EpisodeMeta
from dataset.episode import EpisodeReader
from .action_codec import ActionCodec
from .state_builder import StateBuilder
from .types import ModelLabels


# ---------------------------------------------------------------------------
# Internal index entry — replaces the old eagerly-built ILSample
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _IndexEntry:
    meta: EpisodeMeta
    t: int
    player: int
    value_target: float


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NeuralILDataset(Dataset):
    """Lazy IL dataset — builds samples on demand from an (episode, step, player) index.

    Memory cost: O(N_samples) for the index pointers (~100 bytes each) plus at
    most `reader_cache_size` episodes fully loaded in RAM via EpisodeReader(cache=True).
    No per-sample numpy arrays are kept between __getitem__ calls.

    The index is expected to be ordered by episode so that sequential iteration
    (e.g. DataLoader with shuffle=False) achieves near-100% LRU cache hit rate.
    """

    def __init__(
        self,
        index: list[_IndexEntry],
        state_builder,
        codec,
        use_pointer: bool = False,
        use_planet_policy: bool = False,
        reader_cache_size: int = 64,
    ) -> None:
        self._index = index
        self._state_builder = state_builder
        self._codec = codec
        self._use_pointer = use_pointer
        self._use_planet_policy = use_planet_policy
        self._reader_cache_size = reader_cache_size
        # LRU: path_str -> EpisodeReader with all arrays loaded in memory
        self._reader_cache: OrderedDict[str, EpisodeReader] = OrderedDict()

    @property
    def use_pointer(self) -> bool:
        return self._use_pointer

    def __len__(self) -> int:
        return len(self._index)

    def _get_reader(self, meta: EpisodeMeta) -> EpisodeReader:
        key = str(meta.path)
        if key in self._reader_cache:
            self._reader_cache.move_to_end(key)
            return self._reader_cache[key]
        reader = EpisodeReader(meta, cache=True).__enter__()
        self._reader_cache[key] = reader
        if len(self._reader_cache) > self._reader_cache_size:
            _, evicted = self._reader_cache.popitem(last=False)
            try:
                evicted.__exit__(None, None, None)
            except Exception:
                pass
        return reader

    def __getitem__(self, idx: int) -> dict:
        entry = self._index[idx]
        reader = self._get_reader(entry.meta)
        step = reader.step(entry.t)
        player = entry.player
        value_target = entry.value_target
        raw_actions = step.actions_p0 if player == 0 else step.actions_p1

        if self._use_planet_policy:
            state = self._state_builder.from_step(step, player)
            lv2 = self._codec.encode_per_planet(
                raw_actions,
                state["context"],
                step.planets,
                value_target,
                max_planets=self._state_builder.max_planets,
            )
            return {
                "planet_features": torch.from_numpy(state["planet_features"]).float(),
                "fleet_features": torch.from_numpy(state["fleet_features"]).float(),
                "fleet_mask": torch.from_numpy(state["fleet_mask"]),
                "global_features": torch.from_numpy(state["global_features"]).float(),
                "planet_mask": torch.from_numpy(state["planet_mask"]),
                "action_types": torch.from_numpy(lv2.planet_action_types).long(),
                "target_idxs": torch.from_numpy(lv2.planet_target_idxs).long(),
                "amount_bins": torch.from_numpy(lv2.planet_amount_bins).long(),
                "value_target": torch.tensor(lv2.value_target, dtype=torch.float32),
            }

        if self._use_pointer:
            structured = self._state_builder.from_step_structured(step, player)
            labels = self._codec.encode(
                raw_actions, structured["context"], step.planets, value_target
            )
            return {
                "planet_features": torch.from_numpy(structured["planet_features"]).float(),
                "fleet_features": torch.from_numpy(structured["fleet_features"]).float(),
                "planet_mask": torch.from_numpy(structured["planet_mask"]),
                "action_type": torch.tensor(labels.action_type, dtype=torch.long),
                "source_idx": torch.tensor(labels.source_idx, dtype=torch.long),
                "target_idx": torch.tensor(labels.target_idx, dtype=torch.long),
                "amount_bin": torch.tensor(labels.amount_bin, dtype=torch.long),
                "value_target": torch.tensor(labels.value_target, dtype=torch.float32),
            }

        # Flat model
        model_input = self._state_builder.from_step(step, player)
        labels = self._codec.encode(
            raw_actions, model_input.context, step.planets, value_target
        )
        return {
            "state": torch.from_numpy(model_input.array).float(),
            "action_type": torch.tensor(labels.action_type, dtype=torch.long),
            "source_idx": torch.tensor(labels.source_idx, dtype=torch.long),
            "target_idx": torch.tensor(labels.target_idx, dtype=torch.long),
            "amount_bin": torch.tensor(labels.amount_bin, dtype=torch.long),
            "value_target": torch.tensor(labels.value_target, dtype=torch.float32),
        }


def build_il_dataset(
    catalog: DataCatalog,
    state_builder,
    codec,
    step_filter=None,
    perspective: str = "winner",
    use_pointer: bool = False,
    use_planet_policy: bool = False,
) -> NeuralILDataset:
    """Build a lazy NeuralILDataset from episodes in a DataCatalog.

    Only builds an index of (episode, step, player) tuples — no numpy arrays
    are pre-computed.  When step_filter is provided, each episode file is
    opened once (with array caching) to discover which steps pass the filter.
    The resulting index is ordered by episode so the LRU reader cache in
    __getitem__ achieves near-100% hit rate during sequential iteration.
    """
    if perspective not in ("winner", "loser", "both"):
        raise ValueError(
            f"perspective must be 'winner', 'loser', or 'both'; got {perspective!r}"
        )

    index: list[_IndexEntry] = []

    for meta in catalog.episodes:
        if perspective == "winner":
            if meta.winner == -1:
                continue
            players_to_include = [meta.winner]
        elif perspective == "loser":
            if meta.winner == -1:
                continue
            players_to_include = [1 - meta.winner]
        else:  # "both"
            players_to_include = [0, 1]

        # Capture meta in closure to avoid late-binding inside the loop
        def _value_for(player: int, _meta=meta) -> float:
            if _meta.winner == -1:
                return 0.0
            return 1.0 if _meta.winner == player else -1.0

        if step_filter is not None:
            # Open the file once with array caching for fast per-step filter checks
            with EpisodeReader(meta, cache=True) as reader:
                for t in range(reader.total_steps):
                    step = reader.step(t)
                    if not step_filter(step, meta):
                        continue
                    for player in players_to_include:
                        index.append(_IndexEntry(meta, t, player, _value_for(player)))
        else:
            for t in range(meta.total_steps):
                for player in players_to_include:
                    index.append(_IndexEntry(meta, t, player, _value_for(player)))

    return NeuralILDataset(
        index,
        state_builder,
        codec,
        use_pointer=use_pointer,
        use_planet_policy=use_planet_policy,
    )
