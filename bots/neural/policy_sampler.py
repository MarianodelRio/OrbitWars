"""PolicySampler: samples actions from PlanetPolicyOutput for RL training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .planet_policy_model import PlanetPolicyOutput
from .types import ActionContext

NO_OP = 0
LAUNCH = 1


@dataclass
class CanonicalAction:
    action_types: np.ndarray   # (max_planets,) int8
    target_idxs: np.ndarray    # (max_planets,) int16
    amount_bins: np.ndarray    # (max_planets,) int8


@dataclass
class RLMasks:
    my_planet_mask: torch.BoolTensor   # (max_planets,) or (B, max_planets)
    valid_target_mask: torch.BoolTensor  # (max_planets, max_planets) or (B, max_planets, max_planets)
    planet_mask: torch.BoolTensor      # (max_planets,) or (B, max_planets)


@dataclass
class SampleResult:
    canonical: CanonicalAction
    game_actions: list
    log_prob: torch.Tensor   # scalar
    entropy: torch.Tensor    # scalar
    value: torch.Tensor      # scalar


class PolicySampler:
    def __init__(self, bins: list, max_planets: int) -> None:
        self.bins = bins
        self.max_planets = max_planets

    def build_masks(self, context: ActionContext, device) -> RLMasks:
        P = self.max_planets
        n = context.n_planets

        planet_mask = torch.zeros(P, dtype=torch.bool, device=device)
        planet_mask[:n] = True

        my_planet_mask = torch.zeros(P, dtype=torch.bool, device=device)
        if n > 0:
            ctx_mask = torch.from_numpy(context.my_planet_mask.astype(bool)).to(device)
            my_planet_mask[:n] = ctx_mask[:n]

        valid_target_mask = torch.zeros(P, P, dtype=torch.bool, device=device)
        for i in range(P):
            if planet_mask[i]:
                for j in range(P):
                    if planet_mask[j] and j != i:
                        valid_target_mask[i, j] = True

        return RLMasks(
            my_planet_mask=my_planet_mask,
            valid_target_mask=valid_target_mask,
            planet_mask=planet_mask,
        )

    def sample(
        self,
        output: PlanetPolicyOutput,
        rl_masks: RLMasks,
        context: ActionContext,
        planet_features: np.ndarray,
        deterministic: bool = False,
    ) -> SampleResult:
        P = self.max_planets
        action_types = np.full(P, -1, dtype=np.int8)
        target_idxs = np.full(P, -1, dtype=np.int16)
        amount_bins = np.full(P, -1, dtype=np.int8)

        log_prob = torch.tensor(0.0, dtype=torch.float32)
        entropy = torch.tensor(0.0, dtype=torch.float32)

        mask_np = rl_masks.my_planet_mask.cpu().numpy().astype(bool)  # (P,)
        active_idxs = np.where(mask_np)[0]  # integer indices of my planets
        mask_tensor = rl_masks.my_planet_mask  # bool tensor (P,)

        # Vectorized action-type sampling
        if active_idxs.size > 0:
            at_logits = output.action_type_logits[mask_tensor]  # (n_active, n_at)
            at_dist = Categorical(logits=at_logits)
            if deterministic:
                at_samples = at_logits.argmax(dim=-1)
            else:
                at_samples = at_dist.sample()
            log_prob = log_prob + at_dist.log_prob(at_samples).sum()
            entropy = entropy + at_dist.entropy().sum()
            action_types[active_idxs] = at_samples.cpu().numpy().astype(np.int8)

        # Compute LAUNCH mask
        launch_mask_np = (action_types == LAUNCH)  # (P,) bool numpy
        launch_idxs = np.where(launch_mask_np)[0]
        launch_idxs_tensor = torch.from_numpy(launch_idxs).long().to(output.action_type_logits.device)

        # Vectorized target and amount sampling
        if launch_idxs.size > 0:
            tgt_logits = output.target_logits[launch_idxs_tensor].clone()  # (n_launch, P)
            valid_tgt = rl_masks.valid_target_mask[launch_idxs_tensor]     # (n_launch, P)
            tgt_logits[~valid_tgt] = float("-inf")
            tgt_dist = Categorical(logits=tgt_logits)
            if deterministic:
                tgt_samples = tgt_logits.argmax(dim=-1)
            else:
                tgt_samples = tgt_dist.sample()
            log_prob = log_prob + tgt_dist.log_prob(tgt_samples).sum()
            entropy = entropy + tgt_dist.entropy().sum()
            target_idxs[launch_idxs] = tgt_samples.cpu().numpy().astype(np.int64)

            # Amount
            amt_logits = output.amount_logits[launch_idxs_tensor]  # (n_launch, n_bins)
            amt_dist = Categorical(logits=amt_logits)
            if deterministic:
                amt_samples = amt_logits.argmax(dim=-1)
            else:
                amt_samples = amt_dist.sample()
            log_prob = log_prob + amt_dist.log_prob(amt_samples).sum()
            entropy = entropy + amt_dist.entropy().sum()
            amount_bins[launch_idxs] = amt_samples.cpu().numpy().astype(np.int64)

        canonical = CanonicalAction(
            action_types=action_types,
            target_idxs=target_idxs,
            amount_bins=amount_bins,
        )

        # Build game actions (vectorized)
        game_actions = []
        if launch_idxs.size > 0:
            bins_arr = np.array(self.bins)  # shape (8,), indices 0-7
            safe_amt = np.maximum(0, amount_bins).astype(np.int64)
            safe_amt_clipped = np.minimum(safe_amt, len(bins_arr) - 1)  # clip to 0-7 for indexing

            source_ships_arr = planet_features[:, 5] * 200.0  # (P,)
            fractions = bins_arr[safe_amt_clipped]             # (P,)
            n_ships_arr = fractions * source_ships_arr

            # Combined validity mask
            build_mask = (
                launch_mask_np
                & (target_idxs >= 0)
                & (target_idxs < context.n_planets)
                & (amount_bins >= 0)
                & (amount_bins <= 7)
                & (n_ships_arr >= 1.0)
            )
            build_idxs = np.where(build_mask)[0]

            if build_idxs.size > 0:
                src_pos = np.array([context.planet_positions[i] for i in build_idxs])
                tgt_pos = np.array([context.planet_positions[target_idxs[i]] for i in build_idxs])
                angles = np.arctan2(tgt_pos[:, 1] - src_pos[:, 1], tgt_pos[:, 0] - src_pos[:, 0])
                for k, i in enumerate(build_idxs):
                    game_actions.append([
                        int(context.planet_ids[i]),
                        float(angles[k]),
                        float(n_ships_arr[i]),
                    ])

        # Value: handle (1,) or (1,1)
        value = output.v_outcome.view(-1)[0]

        return SampleResult(
            canonical=canonical,
            game_actions=game_actions,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
        )

    def compute_log_prob(
        self,
        output: PlanetPolicyOutput,
        rl_masks: RLMasks,
        canonical: CanonicalAction,
    ) -> torch.Tensor:
        P = self.max_planets
        device = output.action_type_logits.device
        log_prob = torch.tensor(0.0, dtype=torch.float32, device=device)

        at_tensor = torch.from_numpy(np.array(canonical.action_types)).long().to(device)
        tgt_tensor = torch.from_numpy(np.array(canonical.target_idxs)).long().to(device)
        amt_tensor = torch.from_numpy(np.array(canonical.amount_bins)).long().to(device)

        for i in range(P):
            if not rl_masks.my_planet_mask[i]:
                continue

            at = at_tensor[i].item()
            if at < 0:
                continue

            at_logits = output.action_type_logits[i]
            at_dist = Categorical(logits=at_logits)
            log_prob = log_prob + at_dist.log_prob(at_tensor[i])

            if at == LAUNCH:
                tgt = tgt_tensor[i].item()
                if tgt >= 0:
                    tgt_logits = output.target_logits[i].clone()
                    mask = rl_masks.valid_target_mask[i]
                    tgt_logits[~mask] = float("-inf")
                    tgt_dist = Categorical(logits=tgt_logits)
                    log_prob = log_prob + tgt_dist.log_prob(tgt_tensor[i])

                amt = amt_tensor[i].item()
                if amt >= 0:
                    amt_logits = output.amount_logits[i]
                    amt_dist = Categorical(logits=amt_logits)
                    log_prob = log_prob + amt_dist.log_prob(amt_tensor[i])

        return log_prob

    def compute_entropy(
        self,
        output: PlanetPolicyOutput,
        rl_masks: RLMasks,
    ) -> torch.Tensor:
        P = self.max_planets
        entropy = torch.tensor(0.0, dtype=torch.float32, device=output.action_type_logits.device)

        for i in range(P):
            if not rl_masks.my_planet_mask[i]:
                continue

            at_logits = output.action_type_logits[i]
            at_dist = Categorical(logits=at_logits)
            entropy = entropy + at_dist.entropy()

            tgt_logits = output.target_logits[i].clone()
            mask = rl_masks.valid_target_mask[i]
            tgt_logits[~mask] = float("-inf")
            if mask.any():
                tgt_dist = Categorical(logits=tgt_logits)
                entropy = entropy + tgt_dist.entropy()

            amt_logits = output.amount_logits[i]
            amt_dist = Categorical(logits=amt_logits)
            entropy = entropy + amt_dist.entropy()

        return entropy
