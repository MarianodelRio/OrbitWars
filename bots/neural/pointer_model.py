"""PointerNetworkModel — attention-based policy/value model for Orbit Wars.

Replaces the flat MLP (PolicyValueModel) with a model that has proper inductive
bias for planet selection:
  - Each planet is encoded individually with a shared MLP (planet_encoder).
  - Global state = mean-pool of planet embeddings + projected fleet features.
  - Source head: dot-product attention  score_i = query(global) · key(planet_i)
  - Target head: conditioned pointer    score_i = query(global + source_emb) · key(planet_i)
  - amount / value / action_type heads: linear from global representation.

Expects structured batch tensors (not a flat 1D state vector).  See StateBuilder
and NeuralILDataset for how to produce these.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PointerNetworkConfig:
    planet_input_dim: int = 7
    fleet_input_dim: int = 700      # max_fleets * 7 = 100 * 7
    planet_embed_dim: int = 64
    global_dim: int = 128
    max_planets: int = 50
    max_fleets: int = 100
    n_amount_bins: int = 5
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class PointerPolicyOutput:
    action_type_logits: torch.Tensor  # (B, 2)
    source_logits: torch.Tensor       # (B, max_planets)
    target_logits: torch.Tensor       # (B, max_planets)
    amount_logits: torch.Tensor       # (B, n_amount_bins)
    value: torch.Tensor               # (B, 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PointerNetworkModel(nn.Module):
    """Pointer-network policy/value model.

    Forward inputs:
        planet_features : (B, max_planets, planet_input_dim)
        fleet_features  : (B, fleet_input_dim)
        planet_mask     : (B, max_planets)  bool — True for real planets, False for padding

    Forward output: PointerPolicyOutput
    """

    def __init__(self, config: PointerNetworkConfig) -> None:
        super().__init__()
        self.config = config

        E = config.planet_embed_dim    # 64
        G = config.global_dim          # 128

        # ── Encoders ──────────────────────────────────────────────────────
        self.planet_encoder = nn.Sequential(
            nn.Linear(config.planet_input_dim, E),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(E, E),
            nn.ReLU(),
        )

        self.fleet_proj = nn.Sequential(
            nn.Linear(config.fleet_input_dim, E),
            nn.ReLU(),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(E + E, G),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # ── Pointer heads ─────────────────────────────────────────────────
        self.source_query = nn.Linear(G, E, bias=False)
        self.source_key   = nn.Linear(E, E, bias=False)

        # Target query is conditioned on global + soft source embedding
        self.target_query = nn.Linear(G + E, E, bias=False)
        self.target_key   = nn.Linear(E, E, bias=False)

        # ── Output heads ──────────────────────────────────────────────────
        self.action_type_head = nn.Linear(G, 2)
        self.amount_head      = nn.Linear(G, config.n_amount_bins)
        self.value_head       = nn.Linear(G, 1)

        self._scale = math.sqrt(E)

    # -----------------------------------------------------------------------

    def forward(
        self,
        planet_features: torch.Tensor,
        fleet_features: torch.Tensor,
        planet_mask: torch.Tensor,
    ) -> PointerPolicyOutput:
        """
        planet_features : (B, max_planets, 7)
        fleet_features  : (B, 700)
        planet_mask     : (B, max_planets)  bool — True = real planet
        """
        B, P, _ = planet_features.shape

        # 1. Encode each planet individually with shared MLP
        #    planet_features: (B, P, 7) → reshape to (B*P, 7) → encode → (B, P, E)
        pf_flat = planet_features.view(B * P, -1)
        planet_emb = self.planet_encoder(pf_flat).view(B, P, -1)  # (B, P, E)

        # 2. Masked mean-pool planet embeddings → (B, E)
        #    Expand mask: (B, P) → (B, P, 1)
        mask_f = planet_mask.float().unsqueeze(-1)          # (B, P, 1)
        n_real = mask_f.sum(dim=1).clamp(min=1)             # (B, 1)
        planet_pool = (planet_emb * mask_f).sum(dim=1) / n_real  # (B, E)

        # 3. Encode fleet features → (B, E)
        fleet_emb = self.fleet_proj(fleet_features)         # (B, E)

        # 4. Global representation → (B, G)
        global_repr = self.global_proj(
            torch.cat([planet_pool, fleet_emb], dim=-1)
        )                                                    # (B, G)

        # 5. Source pointer: query over planet keys
        #    q: (B, E)  k: (B, P, E)  logits: (B, P)
        q_src = self.source_query(global_repr)               # (B, E)
        k_src = self.source_key(planet_emb)                  # (B, P, E)
        source_logits = torch.bmm(
            k_src, q_src.unsqueeze(-1)
        ).squeeze(-1) / self._scale                          # (B, P)

        # Mask padding positions to -inf
        source_logits = source_logits.masked_fill(~planet_mask, float("-inf"))

        # 6. Soft source embedding (differentiable) — used to condition target query
        src_weights = torch.softmax(source_logits, dim=-1)   # (B, P)  — NaN-safe: at least 1 real planet
        source_emb = torch.bmm(
            src_weights.unsqueeze(1), planet_emb
        ).squeeze(1)                                         # (B, E)

        # 7. Target pointer: conditioned on global + source_emb
        q_tgt = self.target_query(
            torch.cat([global_repr, source_emb], dim=-1)
        )                                                    # (B, E)
        k_tgt = self.target_key(planet_emb)                  # (B, P, E)
        target_logits = torch.bmm(
            k_tgt, q_tgt.unsqueeze(-1)
        ).squeeze(-1) / self._scale                          # (B, P)
        target_logits = target_logits.masked_fill(~planet_mask, float("-inf"))

        # 8. Other heads from global_repr
        action_type_logits = self.action_type_head(global_repr)   # (B, 2)
        amount_logits      = self.amount_head(global_repr)        # (B, n_bins)
        value              = torch.tanh(self.value_head(global_repr))  # (B, 1)

        return PointerPolicyOutput(
            action_type_logits=action_type_logits,
            source_logits=source_logits,
            target_logits=target_logits,
            amount_logits=amount_logits,
            value=value,
        )
