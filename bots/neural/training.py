"""NeuralILDataset and build_il_dataset — imitation learning dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    state_array: np.ndarray  # (D,) float32 — flat; None when use_pointer=True
    labels: ModelLabels
    planet_features: np.ndarray | None = None  # (max_planets, 7) — pointer mode
    fleet_features: np.ndarray | None = None   # (max_fleets*7,)  — pointer mode
    planet_mask: np.ndarray | None = None      # (max_planets,)   — pointer mode
    # v2 / planet_policy fields
    planet_features_v2: np.ndarray | None = None   # (max_planets, 10) — planet_policy mode
    fleet_features_v2: np.ndarray | None = None    # (max_fleets, 8)   — planet_policy mode
    fleet_mask: np.ndarray | None = None           # (max_fleets,) bool — planet_policy mode
    global_features: np.ndarray | None = None      # (4,) float32      — planet_policy mode
    labels_v2: object | None = None               # PerPlanetLabels   — planet_policy mode


class NeuralILDataset(Dataset):
    def __init__(self, samples: list[ILSample], use_pointer: bool = False, use_planet_policy: bool = False) -> None:
        self._samples = samples
        self._use_pointer = use_pointer
        self._use_planet_policy = use_planet_policy

    @property
    def use_pointer(self) -> bool:
        return self._use_pointer

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]

        if self._use_planet_policy:
            lv2 = sample.labels_v2
            return {
                "planet_features": torch.from_numpy(sample.planet_features_v2).float(),
                "fleet_features": torch.from_numpy(sample.fleet_features_v2).float(),
                "fleet_mask": torch.from_numpy(sample.fleet_mask),
                "global_features": torch.from_numpy(sample.global_features).float(),
                "planet_mask": torch.from_numpy(lv2.my_planet_mask),
                "action_types": torch.from_numpy(lv2.planet_action_types).long(),
                "target_idxs": torch.from_numpy(lv2.planet_target_idxs).long(),
                "amount_bins": torch.from_numpy(lv2.planet_amount_bins).long(),
                "value_target": torch.tensor(lv2.value_target, dtype=torch.float32),
            }

        labels = sample.labels
        label_tensors = {
            "action_type": torch.tensor(labels.action_type, dtype=torch.long),
            "source_idx": torch.tensor(labels.source_idx, dtype=torch.long),
            "target_idx": torch.tensor(labels.target_idx, dtype=torch.long),
            "amount_bin": torch.tensor(labels.amount_bin, dtype=torch.long),
            "value_target": torch.tensor(labels.value_target, dtype=torch.float32),
        }
        if self._use_pointer:
            return {
                "planet_features": torch.from_numpy(sample.planet_features).float(),
                "fleet_features": torch.from_numpy(sample.fleet_features).float(),
                "planet_mask": torch.from_numpy(sample.planet_mask),
                **label_tensors,
            }
        return {
            "state": torch.from_numpy(sample.state_array).float(),
            **label_tensors,
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
    """Build a NeuralILDataset from episodes in a DataCatalog.

    Args:
        catalog: DataCatalog with episode metadata.
        state_builder: StateBuilder for converting steps to model inputs.
        codec: ActionCodec for encoding raw actions to labels.
        step_filter: Optional callable(step, meta) -> bool to skip steps.
        perspective: "winner", "loser", or "both".
        use_pointer: If True, produce structured (planet_features, fleet_features,
            planet_mask) tensors instead of a flat state vector.

    Returns:
        NeuralILDataset ready for DataLoader.
    """
    if perspective not in ("winner", "loser", "both"):
        raise ValueError(
            f"perspective must be 'winner', 'loser', or 'both'; got {perspective!r}"
        )

    samples: list[ILSample] = []

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
                    if use_planet_policy:
                        raw_actions = step.actions_p0 if player == 0 else step.actions_p1
                        state = state_builder.from_step(step, player)
                        labels_v2 = codec.encode_per_planet(
                            raw_actions,
                            state["context"],
                            step.planets,
                            _value_for(player),
                            max_planets=state_builder.max_planets,
                        )
                        samples.append(ILSample(
                            state_array=np.empty(0, dtype=np.float32),
                            labels=ModelLabels(
                                action_type=0,
                                source_idx=-1,
                                target_idx=-1,
                                amount_bin=-1,
                                value_target=labels_v2.value_target,
                            ),
                            planet_features_v2=state["planet_features"],
                            fleet_features_v2=state["fleet_features"],
                            fleet_mask=state["fleet_mask"],
                            global_features=state["global_features"],
                            labels_v2=labels_v2,
                        ))
                    elif use_pointer:
                        structured = state_builder.from_step_structured(step, player)
                        raw_actions = step.actions_p0 if player == 0 else step.actions_p1
                        labels = codec.encode(
                            raw_actions,
                            structured["context"],
                            step.planets,
                            _value_for(player),
                        )
                        samples.append(ILSample(
                            state_array=np.empty(0, dtype=np.float32),
                            labels=labels,
                            planet_features=structured["planet_features"],
                            fleet_features=structured["fleet_features"],
                            planet_mask=structured["planet_mask"],
                        ))
                    else:
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

    return NeuralILDataset(samples, use_pointer=use_pointer, use_planet_policy=use_planet_policy)
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
