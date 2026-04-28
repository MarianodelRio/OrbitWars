"""NeuralILDataset and build_il_dataset — lazy imitation learning dataset pipeline.

NeuralILDataset stores a lightweight index of (episode, step, player) tuples and
builds each sample on demand in __getitem__.  An LRU cache of open EpisodeReaders
(cache=True) amortises HDF5 I/O across all steps of the same episode, keeping
memory bounded regardless of dataset size.

PrecomputedILDataset + build_il_cache provide a fast pre-computed path:
  - build_il_cache: one-time pass over all episodes, writes arrays to HDF5
  - PrecomputedILDataset: __getitem__ is a direct HDF5 slice (~0.1ms vs ~5ms)
Memory-safe: during cache build only one episode is held in RAM at a time;
during training only batch_size samples are in RAM at once.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
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
            encoded = self._codec.encode_per_planet(
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
                "action_types": torch.from_numpy(encoded.planet_action_types).long(),
                "target_idxs": torch.from_numpy(encoded.planet_target_idxs).long(),
                "amount_bins": torch.from_numpy(encoded.planet_amount_bins).long(),
                "value_target": torch.tensor(encoded.value_target, dtype=torch.float32),
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


# ---------------------------------------------------------------------------
# Pre-computed HDF5 cache — fast path for planet_policy IL training
# ---------------------------------------------------------------------------

def build_il_cache(
    catalog: DataCatalog,
    state_builder,
    codec,
    cache_path,
    step_filter=None,
    perspective: str = "winner",
    chunk_size: int = 256,
) -> None:
    """Pre-compute all IL samples and write them to a single HDF5 cache file.

    Memory-safe: only one episode is held in RAM at a time.  Samples are
    written incrementally so the file is valid even if interrupted (though
    incomplete caches are detected and rebuilt by load helpers).

    The HDF5 file stores:
      planet_features  (N, P, 10)  float32
      fleet_features   (N, FL, 8)  float32
      fleet_mask       (N, FL)     uint8
      global_features  (N, 4)      float32
      planet_mask      (N, P)      uint8
      action_types     (N, P)      int8   (-1=pad, 0=NO_OP, 1=LAUNCH)
      target_idxs      (N, P)      int8   (-1=pad/suppressed, 0‥P-1)
      amount_bins      (N, P)      int8   (-1=pad/NO_OP, 0‥4)
      value_target     (N,)        float32
      episode_idx      (N,)        int32  (index into episode_paths dataset)

    Attributes:
      at_counts   (2,)  float32 — action_type label counts across all samples
      amt_counts  (5,)  float32 — amount_bin label counts across all samples
      perspective str
      n_episodes  int

    String dataset:
      episode_paths  (n_episodes,) variable-length UTF-8 — for split reproducibility
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    P = state_builder.max_planets
    FL = state_builder.max_fleets
    # No compression: gzip decompresses the entire chunk (~1.6 MB for fleet_features)
    # on every access, which is slower than the lazy baseline.  Raw HDF5 reads are
    # O(row_bytes / 500 MB/s) ≈ 0.01 ms vs 5 ms for Python computation.
    compress: dict = {}

    at_counts = np.zeros(2, dtype=np.int64)
    amt_counts = np.zeros(5, dtype=np.int64)

    episode_paths = [str(m.path) for m in catalog.episodes]

    with h5py.File(cache_path, "w") as f:
        f.create_dataset(
            "episode_paths",
            data=np.array(episode_paths, dtype=h5py.special_dtype(vlen=str)),
        )

        # Resizable datasets — grow one episode batch at a time
        ds_pf  = f.create_dataset("planet_features",  shape=(0, P, 10),  maxshape=(None, P, 10),  dtype="float32", chunks=(chunk_size, P, 10),  **compress)
        ds_ff  = f.create_dataset("fleet_features",   shape=(0, FL, 8),  maxshape=(None, FL, 8),  dtype="float32", chunks=(chunk_size, FL, 8),  **compress)
        ds_fm  = f.create_dataset("fleet_mask",       shape=(0, FL),     maxshape=(None, FL),     dtype="uint8",   chunks=(chunk_size, FL),      **compress)
        ds_gf  = f.create_dataset("global_features",  shape=(0, 4),      maxshape=(None, 4),      dtype="float32", chunks=(chunk_size, 4),       **compress)
        ds_pm  = f.create_dataset("planet_mask",      shape=(0, P),      maxshape=(None, P),      dtype="uint8",   chunks=(chunk_size, P),       **compress)
        ds_at  = f.create_dataset("action_types",     shape=(0, P),      maxshape=(None, P),      dtype="int8",    chunks=(chunk_size, P),       **compress)
        ds_ti  = f.create_dataset("target_idxs",      shape=(0, P),      maxshape=(None, P),      dtype="int8",    chunks=(chunk_size, P),       **compress)
        ds_ab  = f.create_dataset("amount_bins",      shape=(0, P),      maxshape=(None, P),      dtype="int8",    chunks=(chunk_size, P),       **compress)
        ds_vt  = f.create_dataset("value_target",     shape=(0,),        maxshape=(None,),        dtype="float32", chunks=(chunk_size,),         **compress)
        ds_ei  = f.create_dataset("episode_idx",      shape=(0,),        maxshape=(None,),        dtype="int32",   chunks=(chunk_size,),         **compress)

        all_datasets = [ds_pf, ds_ff, ds_fm, ds_gf, ds_pm, ds_at, ds_ti, ds_ab, ds_vt, ds_ei]
        offset = 0
        n_episodes = len(catalog.episodes)

        for ep_idx, meta in enumerate(catalog.episodes):
            if (ep_idx + 1) % 100 == 0 or ep_idx == 0:
                print(f"  [cache] {ep_idx+1}/{n_episodes} episodes  ({offset:,} samples so far)", flush=True)

            if perspective == "winner":
                if meta.winner == -1:
                    continue
                players = [meta.winner]
            elif perspective == "loser":
                if meta.winner == -1:
                    continue
                players = [1 - meta.winner]
            else:
                players = [0, 1]

            def _value_for(player: int, _meta=meta) -> float:
                if _meta.winner == -1:
                    return 0.0
                return 1.0 if _meta.winner == player else -1.0

            batch_pf, batch_ff, batch_fm, batch_gf, batch_pm = [], [], [], [], []
            batch_at, batch_ti, batch_ab, batch_vt, batch_ei = [], [], [], [], []

            with EpisodeReader(meta, cache=True) as reader:
                for t in range(reader.total_steps):
                    step = reader.step(t)
                    if step_filter is not None and not step_filter(step, meta):
                        continue
                    for player in players:
                        state = state_builder.from_step(step, player)
                        raw_actions = step.actions_p0 if player == 0 else step.actions_p1
                        vt = _value_for(player)
                        encoded = codec.encode_per_planet(
                            raw_actions,
                            state["context"],
                            step.planets,
                            vt,
                            max_planets=state_builder.max_planets,
                        )
                        batch_pf.append(state["planet_features"])
                        batch_ff.append(state["fleet_features"])
                        batch_fm.append(state["fleet_mask"].view(np.uint8))
                        batch_gf.append(state["global_features"])
                        batch_pm.append(state["planet_mask"].view(np.uint8))
                        batch_at.append(encoded.planet_action_types.astype(np.int8))
                        batch_ti.append(encoded.planet_target_idxs.astype(np.int8))
                        batch_ab.append(encoded.planet_amount_bins.astype(np.int8))
                        batch_vt.append(np.float32(vt))
                        batch_ei.append(np.int32(ep_idx))

            if not batch_pf:
                continue

            n = len(batch_pf)
            arrays = [
                np.stack(batch_pf),
                np.stack(batch_ff),
                np.stack(batch_fm),
                np.stack(batch_gf),
                np.stack(batch_pm),
                np.stack(batch_at),
                np.stack(batch_ti),
                np.stack(batch_ab),
                np.array(batch_vt, dtype=np.float32),
                np.array(batch_ei, dtype=np.int32),
            ]
            for ds, arr in zip(all_datasets, arrays):
                ds.resize(offset + n, axis=0)
                ds[offset:offset + n] = arr

            # Accumulate class weight counts (vectorised)
            at_flat = arrays[5].ravel()  # action_types int8
            valid_at = at_flat[(at_flat >= 0) & (at_flat <= 1)].astype(np.intp)
            at_counts += np.bincount(valid_at, minlength=2)

            ab_flat = arrays[7].ravel()  # amount_bins int8
            valid_ab = ab_flat[(ab_flat >= 0) & (ab_flat <= 4)].astype(np.intp)
            amt_counts += np.bincount(valid_ab, minlength=5)

            offset += n

        # Write metadata
        f.attrs["at_counts"] = at_counts.astype(np.float32)
        f.attrs["amt_counts"] = amt_counts.astype(np.float32)
        f.attrs["perspective"] = perspective
        f.attrs["n_episodes"] = n_episodes
        f.attrs["n_samples"] = offset

    print(f"  [cache] done — {offset:,} samples written to {cache_path}", flush=True)


class PrecomputedILDataset(Dataset):
    """Fast IL dataset that reads pre-computed tensors from an HDF5 cache.

    Uses a chunk-buffered read strategy: instead of reading one HDF5 row at a
    time (8 Python API calls per sample ≈ 5ms), loads `_BUFFER_SIZE` samples at
    once via contiguous HDF5 slices (8 calls per 256 samples ≈ 0.02ms/sample).
    The buffer holds ~2 MB of data in RAM, making the approach memory-safe
    regardless of dataset size.

    For sequential access (shuffle=False, default), the buffer hit-rate is
    255/256 = 99.6%, so HDF5 is read only once per 256 samples.

    Args:
        cache_path: path to the HDF5 cache file built by build_il_cache.
        indices: optional int64 array of sample indices within the cache.
                 If None, all samples are used.  Pass a subset for train/val splits.
    """

    _BUFFER_SIZE = 512  # samples per read; trades ~4.5 MB RAM for fewer HDF5 calls

    def __init__(self, cache_path, indices: Optional[np.ndarray] = None) -> None:
        self._path = str(cache_path)
        self._h5: Optional[h5py.File] = None  # opened lazily per-worker

        with h5py.File(self._path, "r") as f:
            self._n_h5 = int(f["planet_features"].shape[0])
            self._at_counts = np.array(f.attrs.get("at_counts", [1.0, 1.0]), dtype=np.float32)
            self._amt_counts = np.array(f.attrs.get("amt_counts", [1.0] * 5), dtype=np.float32)

        self._indices: np.ndarray = (
            np.arange(self._n_h5, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )

        # Chunk buffer — loaded lazily when the first __getitem__ is called.
        # Stores a contiguous HDF5 slice [buf_start : buf_end].
        self._buf_start: int = -1
        self._buf_end: int = -1
        self._buf: dict = {}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def class_weight_counts(self):
        return self._at_counts, self._amt_counts

    def __len__(self) -> int:
        return len(self._indices)

    # ------------------------------------------------------------------
    # HDF5 handle — lazy open, one per process/worker
    # ------------------------------------------------------------------

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self._path, "r")
        return self._h5

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _load_buffer(self, h5_idx: int) -> None:
        """Load a contiguous HDF5 slice containing h5_idx into the in-RAM buffer."""
        B = self._BUFFER_SIZE
        start = (h5_idx // B) * B
        end = min(start + B, self._n_h5)
        h5 = self._get_h5()
        self._buf = {
            "planet_features": h5["planet_features"][start:end],   # (B, P, 10) float32
            "fleet_features":  h5["fleet_features"][start:end],    # (B, FL, 8) float32
            "fleet_mask":      h5["fleet_mask"][start:end],        # (B, FL) uint8
            "global_features": h5["global_features"][start:end],   # (B, 4) float32
            "planet_mask":     h5["planet_mask"][start:end],       # (B, P) uint8
            "action_types":    h5["action_types"][start:end],      # (B, P) int8
            "target_idxs":     h5["target_idxs"][start:end],      # (B, P) int8
            "amount_bins":     h5["amount_bins"][start:end],       # (B, P) int8
            "value_target":    h5["value_target"][start:end],      # (B,) float32
        }
        self._buf_start = start
        self._buf_end = end

    # ------------------------------------------------------------------
    # Core: fast per-sample read from in-RAM buffer
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        h5_idx = int(self._indices[idx])
        if not (self._buf_start <= h5_idx < self._buf_end):
            self._load_buffer(h5_idx)
        loc = h5_idx - self._buf_start
        b = self._buf
        return {
            "planet_features": torch.from_numpy(b["planet_features"][loc].copy()),
            "fleet_features":  torch.from_numpy(b["fleet_features"][loc].copy()),
            "fleet_mask":      torch.from_numpy(b["fleet_mask"][loc].view(np.bool_)),
            "global_features": torch.from_numpy(b["global_features"][loc].copy()),
            "planet_mask":     torch.from_numpy(b["planet_mask"][loc].view(np.bool_)),
            "action_types":    torch.from_numpy(b["action_types"][loc].astype(np.int64)),
            "target_idxs":     torch.from_numpy(b["target_idxs"][loc].astype(np.int64)),
            "amount_bins":     torch.from_numpy(b["amount_bins"][loc].astype(np.int64)),
            "value_target":    torch.tensor(float(b["value_target"][loc]), dtype=torch.float32),
        }


def load_precomputed_split(
    cache_path,
    catalog: DataCatalog,
    train_episodes: list,
    val_episodes: list,
) -> tuple["PrecomputedILDataset", "PrecomputedILDataset"]:
    """Create train/val PrecomputedILDataset views from a single cache file.

    Uses the episode_paths dataset in the cache to map catalog episodes to
    HDF5 episode indices, then filters the episode_idx array to get sample indices
    for each split.  Reads only episode_idx (int32 array) into RAM — O(N_samples)
    but small (~2 MB for 500k samples).
    """
    cache_path = Path(cache_path)
    with h5py.File(cache_path, "r") as f:
        cached_paths = [p.decode() if isinstance(p, bytes) else p for p in f["episode_paths"][:]]
        episode_idx_arr = f["episode_idx"][:]  # (N,) int32

    path_to_cache_ep = {p: i for i, p in enumerate(cached_paths)}

    def _sample_indices(episode_list) -> np.ndarray:
        ep_set = set()
        for meta in episode_list:
            key = str(meta.path)
            if key in path_to_cache_ep:
                ep_set.add(path_to_cache_ep[key])
        if not ep_set:
            return np.array([], dtype=np.int64)
        ep_arr = np.array(sorted(ep_set), dtype=np.int32)
        mask = np.isin(episode_idx_arr, ep_arr)
        return np.where(mask)[0].astype(np.int64)

    train_indices = _sample_indices(train_episodes)
    val_indices = _sample_indices(val_episodes)

    return (
        PrecomputedILDataset(cache_path, indices=train_indices),
        PrecomputedILDataset(cache_path, indices=val_indices),
    )
