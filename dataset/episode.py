"""EpisodeReader and StepRecord — turn-by-turn HDF5 access with padding resolved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from dataset.catalog import EpisodeMeta


@dataclass
class StepRecord:
    turn: int
    planets: np.ndarray          # shape (n_planets, 7) — no padding
    fleets: np.ndarray           # shape (n_fleets, 7) — no padding
    actions_p0: np.ndarray       # shape (n_actions_p0, 3) — no padding
    actions_p1: np.ndarray       # shape (n_actions_p1, 3) — no padding
    comet_planet_ids: np.ndarray # shape (k,) — only real IDs, no -1
    is_terminal: bool


class EpisodeReader:
    def __init__(self, meta: "EpisodeMeta", cache: bool = False) -> None:
        self._meta = meta
        self._cache = cache
        self._file = None
        # Cached arrays (populated in __enter__ when cache=True)
        self._c_planets = None
        self._c_fleets = None
        self._c_actions_p0 = None
        self._c_actions_p1 = None
        self._c_n_planets = None
        self._c_n_fleets = None
        self._c_n_actions_p0 = None
        self._c_n_actions_p1 = None
        self._c_comet_planet_ids = None
        self._c_terminals = None

    def __enter__(self) -> "EpisodeReader":
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required for EpisodeReader") from e

        if self._cache:
            with h5py.File(self._meta.path, "r") as f:
                self._c_planets = f["planets"][:]
                self._c_fleets = f["fleets"][:]
                self._c_actions_p0 = f["actions_p0"][:]
                self._c_actions_p1 = f["actions_p1"][:]
                self._c_n_planets = f["n_planets"][:]
                self._c_n_fleets = f["n_fleets"][:]
                self._c_n_actions_p0 = f["n_actions_p0"][:]
                self._c_n_actions_p1 = f["n_actions_p1"][:]
                self._c_comet_planet_ids = f["comet_planet_ids"][:]
                self._c_terminals = f["terminals"][:]
        else:
            self._file = h5py.File(self._meta.path, "r")

        return self

    def __exit__(self, *_) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def meta(self) -> "EpisodeMeta":
        return self._meta

    @property
    def total_steps(self) -> int:
        return int(self._meta.total_steps)

    def step(self, t: int) -> StepRecord:
        if not (0 <= t < self.total_steps):
            raise IndexError(
                f"Turn {t} out of range [0, {self.total_steps})"
            )

        if self._file is None and self._c_n_planets is None:
            raise RuntimeError("EpisodeReader must be used as a context manager")

        if self._cache:
            n_p = int(self._c_n_planets[t])
            n_f = int(self._c_n_fleets[t])
            n_a0 = int(self._c_n_actions_p0[t])
            n_a1 = int(self._c_n_actions_p1[t])

            planets = self._c_planets[t, :n_p, :].copy()
            fleets = self._c_fleets[t, :n_f, :].copy()
            actions_p0 = self._c_actions_p0[t, :n_a0, :].copy()
            actions_p1 = self._c_actions_p1[t, :n_a1, :].copy()
            comet_row = self._c_comet_planet_ids[t]
            is_terminal = bool(self._c_terminals[t])
        else:
            f = self._file
            n_p = int(f["n_planets"][t])
            n_f = int(f["n_fleets"][t])
            n_a0 = int(f["n_actions_p0"][t])
            n_a1 = int(f["n_actions_p1"][t])

            planets = f["planets"][t, :n_p, :]
            fleets = f["fleets"][t, :n_f, :]
            actions_p0 = f["actions_p0"][t, :n_a0, :]
            actions_p1 = f["actions_p1"][t, :n_a1, :]
            comet_row = f["comet_planet_ids"][t, :]
            is_terminal = bool(f["terminals"][t])

        # Ensure float32 arrays with correct shape even when count is 0
        if n_p == 0:
            planets = np.empty((0, 7), dtype=np.float32)
        if n_f == 0:
            fleets = np.empty((0, 7), dtype=np.float32)
        if n_a0 == 0:
            actions_p0 = np.empty((0, 3), dtype=np.float32)
        if n_a1 == 0:
            actions_p1 = np.empty((0, 3), dtype=np.float32)

        # Filter -1 padding from comet_planet_ids
        comet_ids = np.asarray(comet_row, dtype=np.int32)
        comet_planet_ids = comet_ids[comet_ids != -1]

        return StepRecord(
            turn=t,
            planets=planets,
            fleets=fleets,
            actions_p0=actions_p0,
            actions_p1=actions_p1,
            comet_planet_ids=comet_planet_ids,
            is_terminal=is_terminal,
        )

    def steps(self, start: int = 0, end: int | None = None) -> Iterator[StepRecord]:
        """Yield StepRecords from start to end (exclusive). Never loads more than one turn at a time."""
        if end is None:
            end = self.total_steps
        for t in range(start, end):
            yield self.step(t)

    def all_steps(self) -> list[StepRecord]:
        # Warning: loads the entire episode into memory. Only use when the episode fits in RAM.
        return list(self.steps())
