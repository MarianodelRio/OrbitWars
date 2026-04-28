"""RolloutBuffer: stores rollout transitions and computes GAE."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutStep:
    state: dict              # StructuredStateV2 — dict of numpy arrays
    rl_masks: object         # RLMasks
    canonical: object        # CanonicalAction
    log_prob_old: float
    value: float
    reward: float
    done: bool
    terminal_reward: float
    shaped_reward: float
    player: int
    step_count: int
    advantage: float = 0.0
    ret: float = 0.0


class RolloutBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._steps: list[RolloutStep] = []

    def add(self, step: RolloutStep) -> None:
        if self.is_full():
            raise ValueError("RolloutBuffer is full")
        self._steps.append(step)

    def is_full(self) -> bool:
        return len(self._steps) >= self.capacity

    def clear(self) -> None:
        self._steps = []

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        from training.rl.gae import compute_gae as _compute_gae
        _compute_gae(self._steps, last_value, gamma, gae_lambda)

    def episode_stats(self) -> dict:
        n_episodes = 0
        total_ep_reward = 0.0
        current_ep_reward = 0.0

        for step in self._steps:
            current_ep_reward += step.shaped_reward + step.terminal_reward
            if step.done:
                n_episodes += 1
                total_ep_reward += current_ep_reward
                current_ep_reward = 0.0

        mean_ep_reward = total_ep_reward / n_episodes if n_episodes > 0 else 0.0
        return {
            "n_episodes": n_episodes,
            "mean_ep_reward": mean_ep_reward,
            "total_steps": len(self._steps),
        }

    def get_batches(self, batch_size: int, device: str) -> list[dict]:
        n = len(self._steps)
        indices = list(range(n))
        random.shuffle(indices)

        batches = []
        for start in range(0, n, batch_size):
            batch_indices = indices[start:start + batch_size]
            if not batch_indices:
                continue

            batch_steps = [self._steps[i] for i in batch_indices]

            planet_features = torch.tensor(
                np.stack([s.state["planet_features"] for s in batch_steps]),
                dtype=torch.float32,
                device=device,
            )
            fleet_features = torch.tensor(
                np.stack([s.state["fleet_features"] for s in batch_steps]),
                dtype=torch.float32,
                device=device,
            )
            fleet_mask = torch.tensor(
                np.stack([s.state["fleet_mask"] for s in batch_steps]),
                dtype=torch.bool,
                device=device,
            )
            planet_mask = torch.tensor(
                np.stack([s.state["planet_mask"] for s in batch_steps]),
                dtype=torch.bool,
                device=device,
            )
            global_features = torch.tensor(
                np.stack([s.state["global_features"] for s in batch_steps]),
                dtype=torch.float32,
                device=device,
            )

            # RLMasks — each is a tensor
            my_planet_mask = torch.stack([s.rl_masks.my_planet_mask.cpu() for s in batch_steps]).to(device)
            valid_target_mask = torch.stack([s.rl_masks.valid_target_mask.cpu() for s in batch_steps]).to(device)
            planet_mask_rl = torch.stack([s.rl_masks.planet_mask.cpu() for s in batch_steps]).to(device)

            action_types = torch.tensor(
                np.stack([s.canonical.action_types.astype(np.int64) for s in batch_steps]),
                dtype=torch.int64,
                device=device,
            )
            target_idxs = torch.tensor(
                np.stack([s.canonical.target_idxs.astype(np.int64) for s in batch_steps]),
                dtype=torch.int64,
                device=device,
            )
            amount_bins = torch.tensor(
                np.stack([s.canonical.amount_bins.astype(np.int64) for s in batch_steps]),
                dtype=torch.int64,
                device=device,
            )

            log_prob_old = torch.tensor(
                [s.log_prob_old for s in batch_steps],
                dtype=torch.float32,
                device=device,
            )
            value_old = torch.tensor(
                [s.value for s in batch_steps],
                dtype=torch.float32,
                device=device,
            )
            advantage = torch.tensor(
                [s.advantage for s in batch_steps],
                dtype=torch.float32,
                device=device,
            )
            ret = torch.tensor(
                [s.ret for s in batch_steps],
                dtype=torch.float32,
                device=device,
            )

            batches.append({
                "planet_features": planet_features,
                "fleet_features": fleet_features,
                "fleet_mask": fleet_mask,
                "planet_mask": planet_mask,
                "global_features": global_features,
                "my_planet_mask": my_planet_mask,
                "valid_target_mask": valid_target_mask,
                "planet_mask_rl": planet_mask_rl,
                "action_types": action_types,
                "target_idxs": target_idxs,
                "amount_bins": amount_bins,
                "log_prob_old": log_prob_old,
                "value_old": value_old,
                "advantage": advantage,
                "ret": ret,
            })

        return batches
