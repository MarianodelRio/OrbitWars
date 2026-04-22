"""DataCatalog and EpisodeMeta — episode discovery and filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ROOTS = [
    _REPO_ROOT / "data" / "matches",
    _REPO_ROOT / "data" / "tournaments",
]


@dataclass
class EpisodeMeta:
    path: Path
    bot0: str
    bot1: str
    winner: int          # 0, 1, or -1 (draw)
    done_reason: str     # "step_limit" | "elimination"
    total_steps: int
    final_ships_p0: float
    final_ships_p1: float


class DataCatalog:
    def __init__(self, episodes: list[EpisodeMeta]) -> None:
        self._episodes = episodes

    @classmethod
    def scan(cls, roots: list[Path] | None = None) -> "DataCatalog":
        """Discover all .h5 episodes under roots, reading only file attrs."""
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required for DataCatalog.scan()") from e

        if roots is None:
            roots = _DEFAULT_ROOTS

        episodes: list[EpisodeMeta] = []
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.h5"):
                try:
                    with h5py.File(path, "r") as f:
                        attrs = f.attrs
                        meta = EpisodeMeta(
                            path=path,
                            bot0=str(attrs["bot0"]),
                            bot1=str(attrs["bot1"]),
                            winner=int(attrs["winner"]),
                            done_reason=str(attrs["done_reason"]),
                            total_steps=int(attrs["total_steps"]),
                            final_ships_p0=float(attrs["final_ships_p0"]),
                            final_ships_p1=float(attrs["final_ships_p1"]),
                        )
                    episodes.append(meta)
                except Exception as exc:
                    print(f"[DataCatalog] Skipping {path}: {exc}")

        return cls(episodes)

    def filter(
        self,
        bot: str | None = None,
        opponent: str | None = None,
        winner_only: bool = False,
        done_reason: str | None = None,
        min_steps: int | None = None,
        max_steps: int | None = None,
    ) -> "DataCatalog":
        """Return a new DataCatalog with the filtered subset."""
        if winner_only and bot is None:
            raise ValueError("winner_only=True requires bot to be specified")

        filtered = []
        for meta in self._episodes:
            # bot filter: bot0 or bot1 contains the string
            if bot is not None:
                in_p0 = bot in meta.bot0
                in_p1 = bot in meta.bot1
                if not (in_p0 or in_p1):
                    continue

                # winner_only: normalize perspective
                if winner_only:
                    if in_p0 and in_p1:
                        # self-play: accept if someone won
                        if meta.winner == -1:
                            continue
                    elif in_p0:
                        if meta.winner != 0:
                            continue
                    else:  # in_p1 only
                        if meta.winner != 1:
                            continue

                # opponent filter
                if opponent is not None:
                    if in_p0:
                        if opponent not in meta.bot1:
                            continue
                    else:
                        if opponent not in meta.bot0:
                            continue
            elif opponent is not None:
                # opponent without bot: either side
                if opponent not in meta.bot0 and opponent not in meta.bot1:
                    continue

            if done_reason is not None and meta.done_reason != done_reason:
                continue
            if min_steps is not None and meta.total_steps < min_steps:
                continue
            if max_steps is not None and meta.total_steps > max_steps:
                continue

            filtered.append(meta)

        return DataCatalog(filtered)

    @property
    def episodes(self) -> list[EpisodeMeta]:
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"DataCatalog({len(self._episodes)} episodes)"

    def save_index(self, path: Path) -> None:
        # Ciclo B
        raise NotImplementedError("save_index not implemented — Ciclo B")

    @classmethod
    def load_index(cls, path: Path) -> "DataCatalog":
        # Ciclo B
        raise NotImplementedError("load_index not implemented — Ciclo B")
