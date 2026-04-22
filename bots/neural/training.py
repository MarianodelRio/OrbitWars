"""NeuralILDataset and build_il_dataset — imitation learning dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.catalog import DataCatalog
from dataset.episode import EpisodeReader
from .action_codec import ActionCodec
from .state_builder import StateBuilder
from .types import ModelLabels


@dataclass
class ILSample:
    state_array: np.ndarray  # (D,) float32
    labels: ModelLabels


class NeuralILDataset(Dataset):
    def __init__(self, samples: list[ILSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        labels = sample.labels
        return {
            "state": torch.from_numpy(sample.state_array).float(),
            "action_type": torch.tensor(labels.action_type, dtype=torch.long),
            "source_idx": torch.tensor(labels.source_idx, dtype=torch.long),
            "target_idx": torch.tensor(labels.target_idx, dtype=torch.long),
            "amount_bin": torch.tensor(labels.amount_bin, dtype=torch.long),
            "value_target": torch.tensor(labels.value_target, dtype=torch.float32),
        }


def build_il_dataset(
    catalog: DataCatalog,
    state_builder: StateBuilder,
    codec: ActionCodec,
    step_filter=None,
    perspective: str = "winner",
) -> NeuralILDataset:
    """Build a NeuralILDataset from episodes in a DataCatalog.

    Args:
        catalog: DataCatalog with episode metadata.
        state_builder: StateBuilder for converting steps to model inputs.
        codec: ActionCodec for encoding raw actions to labels.
        step_filter: Optional callable(step, meta) -> bool to skip steps.
        perspective: "winner", "loser", or "both" — controls which player's
            actions are included. Draws are skipped for "winner" and "loser".

    Returns:
        NeuralILDataset ready for DataLoader.
    """
    if perspective not in ("winner", "loser", "both"):
        raise ValueError(
            f"perspective must be 'winner', 'loser', or 'both'; got {perspective!r}"
        )

    samples: list[ILSample] = []

    for meta in catalog.episodes:
        # Determine which players to include based on perspective
        if perspective == "winner":
            if meta.winner == -1:
                continue  # draw — skip
            players_to_include = [meta.winner]
        elif perspective == "loser":
            if meta.winner == -1:
                continue  # draw — skip
            players_to_include = [1 - meta.winner]
        else:  # "both"
            players_to_include = [0, 1]

        def _value_for(player: int) -> float:
            if meta.winner == -1:
                return 0.0
            return 1.0 if meta.winner == player else -1.0

        with EpisodeReader(meta, cache=True) as reader:
            for t in range(reader.total_steps):
                step = reader.step(t)

                if step_filter is not None and not step_filter(step, meta):
                    continue

                for player in players_to_include:
                    model_input = state_builder.from_step(step, player)
                    raw_actions = step.actions_p0 if player == 0 else step.actions_p1
                    labels = codec.encode(
                        raw_actions,
                        model_input.context,
                        step.planets,
                        _value_for(player),
                    )
                    samples.append(ILSample(
                        state_array=model_input.array,
                        labels=labels,
                    ))

    return NeuralILDataset(samples)
