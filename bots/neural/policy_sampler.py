"""PolicySampler: samples actions from PlanetPolicyOutput for RL training."""

from __future__ import annotations

import math
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

        # output has shapes (max_planets, 2), (max_planets, max_planets), (max_planets, 5), (1,) or (1,1)
        for i in range(P):
            if not rl_masks.my_planet_mask[i]:
                continue

            at_logits = output.action_type_logits[i]  # (2,)
            at_dist = Categorical(logits=at_logits)
            if deterministic:
                at = int(at_logits.argmax().item())
            else:
                at = int(at_dist.sample().item())

            action_types[i] = at
            log_prob = log_prob + at_dist.log_prob(torch.tensor(at, dtype=torch.long))
            entropy = entropy + at_dist.entropy()

            if at == LAUNCH:
                tgt_logits = output.target_logits[i].clone()  # (max_planets,)
                mask = rl_masks.valid_target_mask[i]  # (max_planets,)
                tgt_logits[~mask] = float("-inf")

                tgt_dist = Categorical(logits=tgt_logits)
                if deterministic:
                    tgt = int(tgt_logits.argmax().item())
                else:
                    tgt = int(tgt_dist.sample().item())

                target_idxs[i] = tgt
                log_prob = log_prob + tgt_dist.log_prob(torch.tensor(tgt, dtype=torch.long))
                entropy = entropy + tgt_dist.entropy()

                amt_logits = output.amount_logits[i]  # (n_bins,)
                amt_dist = Categorical(logits=amt_logits)
                if deterministic:
                    amt = int(amt_logits.argmax().item())
                else:
                    amt = int(amt_dist.sample().item())

                amount_bins[i] = amt
                log_prob = log_prob + amt_dist.log_prob(torch.tensor(amt, dtype=torch.long))
                entropy = entropy + amt_dist.entropy()

        canonical = CanonicalAction(
            action_types=action_types,
            target_idxs=target_idxs,
            amount_bins=amount_bins,
        )

        # Build game actions
        game_actions = []
        for i in range(P):
            if action_types[i] != LAUNCH:
                continue
            tgt = int(target_idxs[i])
            if tgt < 0 or tgt >= context.n_planets:
                continue
            amt = int(amount_bins[i])
            if amt < 0 or amt >= len(self.bins):
                continue
            source_ships = float(planet_features[i, 5]) * 200.0
            n_ships = self.bins[amt] * source_ships
            if n_ships < 1.0:
                continue
            source_pos = context.planet_positions[i]
            target_pos = context.planet_positions[tgt]
            angle = math.atan2(
                float(target_pos[1]) - float(source_pos[1]),
                float(target_pos[0]) - float(source_pos[0]),
            )
            game_actions.append([int(context.planet_ids[i]), angle, n_ships])

        # Value: handle (1,) or (1,1)
        value = output.value.view(-1)[0]

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
