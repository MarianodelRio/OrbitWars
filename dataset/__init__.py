"""dataset — data access layer for Orbit Wars training.

Converts raw HDF5 match files into clean, training-ready data structures.
Supports Imitation Learning, offline RL, and future online RL reuse.

Layers:
  catalog   — discover and filter episodes from HDF5 attrs
  episode   — read episodes turn-by-turn with padding resolved
  builder   — build training samples (Ciclo B)
  transforms — pluggable state/action/reward/filter callables
  torch_adapter — PyTorch Dataset wrappers (Ciclo B)
"""

from dataset.catalog import DataCatalog, EpisodeMeta
