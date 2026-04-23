"""PlanetPolicyModel — per-planet entity-centric policy/value model for Orbit Wars."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PlanetPolicyConfig:
    Dp: int = 10             # planet feature dim
    Df: int = 8              # fleet feature dim
    Dg: int = 4              # global feature dim
    E: int = 64              # planet embed dim
    F: int = 32              # fleet embed dim
    G: int = 128             # global repr dim
    max_planets: int = 50
    max_fleets: int = 200
    n_amount_bins: int = 5
    dropout: float = 0.1
    n_attn_heads: int = 2


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class PlanetPolicyOutput:
    action_type_logits: torch.Tensor  # (B, max_planets, 2)
    target_logits: torch.Tensor       # (B, max_planets, max_planets)
    amount_logits: torch.Tensor       # (B, max_planets, n_amount_bins)
    value: torch.Tensor               # (B, 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PlanetPolicyModel(nn.Module):
    """Per-planet entity-centric policy/value model.

    Forward inputs:
        planet_features : (B, max_planets, Dp)
        fleet_features  : (B, max_fleets, Df)
        fleet_mask      : (B, max_fleets)   bool — True for real fleets
        global_features : (B, Dg)
        planet_mask     : (B, max_planets)  bool — True for real planets

    Forward output: PlanetPolicyOutput
    """

    def __init__(self, config: PlanetPolicyConfig) -> None:
        super().__init__()
        self.config = config

        Dp = config.Dp
        Df = config.Df
        Dg = config.Dg
        E = config.E
        F = config.F
        G = config.G
        n_amount_bins = config.n_amount_bins
        n_attn_heads = config.n_attn_heads

        # Encoders
        self.planet_encoder = nn.Sequential(
            nn.Linear(Dp, E),
            nn.ReLU(),
            nn.Linear(E, E),
            nn.ReLU(),
        )

        self.fleet_encoder = nn.Sequential(
            nn.Linear(Df, F),
            nn.ReLU(),
        )

        # Self-attention over planets
        self.planet_attn = nn.MultiheadAttention(
            embed_dim=E,
            num_heads=n_attn_heads,
            batch_first=True,
            dropout=config.dropout,
        )
        self.planet_attn_norm = nn.LayerNorm(E)

        # Global MLP: planet_pool (E) + fleet_ctx (F) + global_features (Dg) → G
        self.global_mlp = nn.Sequential(
            nn.Linear(E + F + Dg, G),
            nn.ReLU(),
            nn.Linear(G, G),
        )

        # Per-planet heads (operate on h = [planet_ctx, global_repr_expanded] of dim E+G)
        self.action_type_head = nn.Linear(E + G, 2)
        self.amount_head = nn.Linear(E + G, n_amount_bins)

        # Pointer network heads
        self.W_query = nn.Linear(E + G, E, bias=False)   # per-planet query
        self.W_key = nn.Linear(E, E, bias=False)          # shared key projection

        # Value head from global repr
        self.value_head = nn.Linear(G, 1)

        self._scale = math.sqrt(E)

    def forward(
        self,
        planet_features: torch.Tensor,
        fleet_features: torch.Tensor,
        fleet_mask: torch.Tensor,
        global_features: torch.Tensor,
        planet_mask: torch.Tensor,
    ) -> PlanetPolicyOutput:
        """
        planet_features : (B, max_planets, Dp)
        fleet_features  : (B, max_fleets, Df)
        fleet_mask      : (B, max_fleets)   bool — True = real fleet
        global_features : (B, Dg)
        planet_mask     : (B, max_planets)  bool — True = real planet
        """
        B, P, _ = planet_features.shape

        # Stage 1: Encode each planet
        planet_emb = self.planet_encoder(
            planet_features.view(B * P, -1)
        ).view(B, P, self.config.E)   # (B, P, E)

        # Stage 2: Encode fleets and masked mean-pool
        fleet_emb_all = self.fleet_encoder(fleet_features)   # (B, max_fleets, F)
        fleet_mask_f = fleet_mask.float().unsqueeze(-1)       # (B, max_fleets, 1)
        n_real_f = fleet_mask_f.sum(dim=1).clamp(min=1)       # (B, 1)
        fleet_ctx = (fleet_emb_all * fleet_mask_f).sum(dim=1) / n_real_f  # (B, F)

        # Stage 3: Self-attention over planets
        # key_padding_mask: True means "ignore" in nn.MultiheadAttention
        attn_out, _ = self.planet_attn(
            planet_emb, planet_emb, planet_emb,
            key_padding_mask=~planet_mask,
        )   # (B, P, E)
        planet_ctx = self.planet_attn_norm(planet_emb + attn_out)   # (B, P, E)

        # Stage 4: Global representation
        planet_mask_f = planet_mask.float().unsqueeze(-1)       # (B, P, 1)
        n_real_p = planet_mask_f.sum(dim=1).clamp(min=1)         # (B, 1)
        planet_pool = (planet_ctx * planet_mask_f).sum(dim=1) / n_real_p  # (B, E)

        global_input = torch.cat([planet_pool, fleet_ctx, global_features], dim=-1)  # (B, E+F+Dg)
        global_repr = self.global_mlp(global_input)   # (B, G)

        # Stage 5: Per-planet outputs
        # h: (B, P, E+G) — per-planet context concatenated with global
        h = torch.cat(
            [planet_ctx, global_repr.unsqueeze(1).expand(-1, P, -1)],
            dim=-1,
        )   # (B, P, E+G)

        action_type_logits = self.action_type_head(h)   # (B, P, 2)
        amount_logits = self.amount_head(h)              # (B, P, n_amount_bins)

        # Pointer: queries from h, keys from planet_ctx
        queries = self.W_query(h)                        # (B, P, E)
        keys = self.W_key(planet_ctx)                    # (B, P, E)
        target_logits = torch.bmm(queries, keys.transpose(1, 2)) / self._scale  # (B, P, P)

        # Value head from global repr
        value = torch.tanh(self.value_head(global_repr))   # (B, 1)

        return PlanetPolicyOutput(
            action_type_logits=action_type_logits,
            target_logits=target_logits,
            amount_logits=amount_logits,
            value=value,
        )
