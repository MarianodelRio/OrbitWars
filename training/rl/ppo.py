"""PPO loss computation for Orbit Wars RL training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from bots.neural.planet_policy_model import PlanetPolicyModel, PlanetPolicyOutput


@dataclass
class PPOLossResult:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    kl_bc: float = 0.0
    entropy_action_type: float = 0.0
    entropy_target: float = 0.0
    entropy_amount: float = 0.0
    kl_bc_action_type: float = 0.0
    kl_bc_target: float = 0.0
    kl_bc_amount: float = 0.0


def _batched_log_prob_and_entropy(
    output: PlanetPolicyOutput,
    my_planet_mask: torch.Tensor,       # (B, P) bool
    valid_target_mask: torch.Tensor,    # (B, P, P) bool
    action_types: torch.Tensor,         # (B, P) int64
    target_idxs: torch.Tensor,          # (B, P) int64
    amount_bins: torch.Tensor,          # (B, P) int64
):
    """Batched differentiable log-prob and entropy computation using torch.gather."""
    B, P, _ = output.action_type_logits.shape
    device = output.action_type_logits.device

    # --- Action type ---
    at_logits = output.action_type_logits  # (B, P, 2)
    at_log_probs_all = F.log_softmax(at_logits, dim=-1)  # (B, P, 2)
    # Clamp action_types to [0,1] for gather (padding slots have -1)
    at_clamped = action_types.clamp(min=0)  # (B, P)
    at_lp = at_log_probs_all.gather(-1, at_clamped.unsqueeze(-1)).squeeze(-1)  # (B, P)
    at_probs = F.softmax(at_logits, dim=-1)
    at_entropy = -(at_probs * at_log_probs_all).sum(-1)  # (B, P)

    # --- Target ---
    tgt_logits = output.target_logits.clone()  # (B, P, P)
    # Apply valid_target_mask: set invalid to -inf
    tgt_logits = tgt_logits.masked_fill(~valid_target_mask, float("-inf"))
    # Guard: if no valid target exists for a slot, set at least one to 0 to avoid NaN in log_softmax
    has_valid_target = valid_target_mask.any(dim=-1)  # (B, P)
    safe_tgt_logits = tgt_logits.clone()
    safe_tgt_logits[:, :, 0] = torch.where(
        ~has_valid_target,
        torch.zeros_like(safe_tgt_logits[:, :, 0]),
        tgt_logits[:, :, 0],
    )
    tgt_log_probs_all = F.log_softmax(safe_tgt_logits, dim=-1)  # (B, P, P)
    tgt_clamped = target_idxs.clamp(min=0)  # (B, P)
    tgt_lp = tgt_log_probs_all.gather(-1, tgt_clamped.unsqueeze(-1)).squeeze(-1)  # (B, P)
    tgt_probs = F.softmax(safe_tgt_logits, dim=-1)
    tgt_entropy = -(tgt_probs * tgt_log_probs_all.clamp(min=-1e9)).sum(-1)  # (B, P)
    # Zero out log_prob and entropy for slots with no valid targets
    tgt_lp = torch.where(has_valid_target, tgt_lp, torch.zeros_like(tgt_lp))
    tgt_entropy = torch.where(has_valid_target, tgt_entropy, torch.zeros_like(tgt_entropy))

    # --- Amount ---
    amt_logits = output.amount_logits  # (B, P, n_bins)
    amt_log_probs_all = F.log_softmax(amt_logits, dim=-1)
    amt_clamped = amount_bins.clamp(min=0)
    amt_lp = amt_log_probs_all.gather(-1, amt_clamped.unsqueeze(-1)).squeeze(-1)  # (B, P)
    amt_probs = F.softmax(amt_logits, dim=-1)
    amt_entropy = -(amt_probs * amt_log_probs_all).sum(-1)  # (B, P)

    # --- Combine ---
    # Only accumulate for my-planet slots where action_types != -1 (not padding)
    is_my_planet = my_planet_mask  # (B, P)
    is_launch = (action_types == 1) & is_my_planet  # (B, P)
    is_noop = (action_types == 0) & is_my_planet    # (B, P)

    # log_prob per item in batch
    # Sum action_type log_prob for all my planets
    log_prob = (at_lp * is_my_planet.float()).sum(-1)  # (B,)
    # Add target and amount log_prob for LAUNCH planets
    log_prob = log_prob + (tgt_lp * is_launch.float()).sum(-1)
    log_prob = log_prob + (amt_lp * is_launch.float()).sum(-1)

    # per-head entropy sums per item
    at_entropy_sum = (at_entropy * is_my_planet.float()).sum(-1)   # (B,)
    tgt_entropy_sum = (tgt_entropy * is_launch.float()).sum(-1)    # (B,)
    amt_entropy_sum = (amt_entropy * is_launch.float()).sum(-1)    # (B,)

    return log_prob, at_entropy_sum, tgt_entropy_sum, amt_entropy_sum  # each (B,)


def compute_ppo_loss(
    model: PlanetPolicyModel,
    batch: dict,
    config,
    bc_model=None,
    kl_bc_coef: float = 0.0,
) -> tuple[torch.Tensor, PPOLossResult]:
    """Compute PPO loss for a batch. Returns (total_loss_tensor, PPOLossResult)."""
    device = next(model.parameters()).device

    planet_features = batch["planet_features"]
    fleet_features = batch["fleet_features"]
    fleet_mask = batch["fleet_mask"]
    global_features = batch["global_features"]
    planet_mask = batch["planet_mask"]

    my_planet_mask = batch["my_planet_mask"]
    valid_target_mask = batch["valid_target_mask"]
    action_types = batch["action_types"]
    target_idxs = batch["target_idxs"]
    amount_bins = batch["amount_bins"]

    log_prob_old = batch["log_prob_old"]    # (B,)
    value_old = batch["value_old"]          # (B,)
    advantage = batch["advantage"]          # (B,)
    ret = batch["ret"]                      # (B,)

    if config.normalize_advantages:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    rt = batch.get("relational_tensor")
    output, _ = model(planet_features, fleet_features, fleet_mask, global_features, planet_mask, rt)

    # Batched log-prob and per-head entropy
    new_log_prob, at_ent, tgt_ent, amt_ent = _batched_log_prob_and_entropy(
        output=output,
        my_planet_mask=my_planet_mask,
        valid_target_mask=valid_target_mask,
        action_types=action_types,
        target_idxs=target_idxs,
        amount_bins=amount_bins,
    )

    # Policy loss
    ratio = torch.exp(new_log_prob - log_prob_old)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantage
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (clipped)
    value_pred = output.v_shaped.squeeze(-1).squeeze(-1)  # (B,)
    v_clipped = value_old + torch.clamp(value_pred - value_old, -config.clip_eps, config.clip_eps)
    value_loss = 0.5 * torch.max(
        (value_pred - ret) ** 2,
        (v_clipped - ret) ** 2,
    ).mean()

    # Per-head weighted entropy bonus
    _at_ent_mean = at_ent.mean().item()
    _tgt_ent_mean = tgt_ent.mean().item()
    _amt_ent_mean = amt_ent.mean().item()
    entropy_bonus = (
        config.entropy_coef_action_type * at_ent.mean()
        + config.entropy_coef_target * tgt_ent.mean()
        + config.entropy_coef_amount * amt_ent.mean()
    )
    entropy_for_log = (at_ent + tgt_ent + amt_ent).mean()

    total_loss = policy_loss + config.vf_coef * value_loss - entropy_bonus

    # KL-to-BC regularization
    _kl_at_val = 0.0
    _kl_tgt_val = 0.0
    _kl_amt_val = 0.0
    kl_bc_val = 0.0
    if bc_model is not None and kl_bc_coef > 0.0:
        is_my_planet = my_planet_mask           # (B, P)
        is_launch = (action_types == 1) & is_my_planet  # (B, P)

        with torch.no_grad():
            bc_out, _ = bc_model(
                planet_features, fleet_features, fleet_mask,
                global_features, planet_mask,
                rt, None
            )

        # KL for action_type head
        kl_at = F.kl_div(
            F.log_softmax(output.action_type_logits, dim=-1),
            F.softmax(bc_out.action_type_logits, dim=-1),
            reduction="none",
        ).sum(-1)  # (B, P)
        kl_at = (kl_at * is_my_planet.float()).sum() / is_my_planet.float().sum().clamp(min=1)

        # KL for target head (apply valid_target_mask before softmax)
        tgt_logits = output.target_logits.clone()
        bc_tgt_logits = bc_out.target_logits.clone()
        if "valid_target_mask" in batch:
            tgt_logits = tgt_logits.masked_fill(~batch["valid_target_mask"], float("-inf"))
            bc_tgt_logits = bc_tgt_logits.masked_fill(~batch["valid_target_mask"], float("-inf"))
        kl_tgt = F.kl_div(
            F.log_softmax(tgt_logits, dim=-1),
            F.softmax(bc_tgt_logits, dim=-1),
            reduction="none",
        ).sum(-1)  # (B, P)
        kl_tgt = (kl_tgt * is_launch.float()).sum() / is_launch.float().sum().clamp(min=1)

        # KL for amount head
        kl_amt = F.kl_div(
            F.log_softmax(output.amount_logits, dim=-1),
            F.softmax(bc_out.amount_logits, dim=-1),
            reduction="none",
        ).sum(-1)  # (B, P)
        kl_amt = (kl_amt * is_launch.float()).sum() / is_launch.float().sum().clamp(min=1)

        _kl_at_val = kl_at.item()
        _kl_tgt_val = kl_tgt.item()
        _kl_amt_val = kl_amt.item()
        kl_bc_total = kl_at + kl_tgt + kl_amt
        total_loss = total_loss + kl_bc_coef * kl_bc_total
        kl_bc_val = kl_bc_total.item()

    # Diagnostics (detached)
    with torch.no_grad():
        approx_kl = (log_prob_old - new_log_prob).mean().item()
        clip_fraction = ((ratio - 1.0).abs() > config.clip_eps).float().mean().item()
        var_ret = ret.var()
        explained_variance = (
            (1.0 - (ret - value_pred).var() / var_ret).item()
            if var_ret > 1e-8
            else 0.0
        )

    result = PPOLossResult(
        total_loss=total_loss.item(),
        policy_loss=policy_loss.item(),
        value_loss=value_loss.item(),
        entropy=entropy_for_log.item(),
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        explained_variance=explained_variance,
        kl_bc=kl_bc_val,
        entropy_action_type=_at_ent_mean,
        entropy_target=_tgt_ent_mean,
        entropy_amount=_amt_ent_mean,
        kl_bc_action_type=_kl_at_val,
        kl_bc_target=_kl_tgt_val,
        kl_bc_amount=_kl_amt_val,
    )

    return total_loss, result
