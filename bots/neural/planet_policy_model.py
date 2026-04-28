"""PlanetPolicyModel — per-planet entity-centric policy/value model for Orbit Wars."""

from __future__ import annotations

from typing import Optional, Tuple
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PlanetPolicyConfig:
    Dp: int = 24
    Df: int = 16
    Dg: int = 16
    E: int = 192
    F: int = 128
    G: int = 384
    n_heads: int = 8
    n_layers: int = 4
    ffn_hidden: int = 768
    dropout: float = 0.1
    max_planets: int = 50
    max_fleets: int = 200
    n_amount_bins: int = 8
    lstm_bypass: bool = False


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class PlanetPolicyOutput:
    action_type_logits: torch.Tensor   # (B, P, 3)
    target_logits: torch.Tensor        # (B, P, P)
    amount_logits: torch.Tensor        # (B, P, n_amount_bins)
    v_outcome: torch.Tensor            # (B, 1)  tanh-bounded
    v_score_diff: torch.Tensor         # (B, 1)  unbounded
    v_shaped: torch.Tensor             # (B, 1)  unbounded
    aux_outcome: Optional[torch.Tensor] = None        # (B, 1)
    aux_return_10: Optional[torch.Tensor] = None      # (B, 1)
    aux_return_50: Optional[torch.Tensor] = None      # (B, 1)
    aux_ownership_10: Optional[torch.Tensor] = None   # (B, P)
    aux_opponent_launch: Optional[torch.Tensor] = None  # (B, P)


# ---------------------------------------------------------------------------
# Inner block
# ---------------------------------------------------------------------------

class PlanetBlock(nn.Module):
    def __init__(self, E, n_heads, ffn_hidden, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(E)
        self.attn = nn.MultiheadAttention(embed_dim=E, num_heads=n_heads,
                                           batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(E)
        self.ffn = nn.Sequential(
            nn.Linear(E, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, E),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        # Pre-LN self-attention
        x_ln = self.ln1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln,
                                 attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask,
                                 need_weights=False)
        x = x + attn_out
        # Pre-LN FFN
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PlanetPolicyModel(nn.Module):
    def __init__(self, config: PlanetPolicyConfig = None):
        super().__init__()
        if config is None:
            config = PlanetPolicyConfig()
        self.config = config
        E, F, G = config.E, config.F, config.G
        Dp, Df, Dg = config.Dp, config.Df, config.Dg

        # Stage 0: encoders
        self.planet_encoder = nn.Sequential(
            nn.Linear(Dp, E), nn.GELU(), nn.Linear(E, E), nn.LayerNorm(E)
        )
        self.fleet_encoder = nn.Sequential(
            nn.Linear(Df, F), nn.GELU(), nn.Linear(F, F), nn.LayerNorm(F)
        )

        # Stage 1: cross-attention fleets→planets
        self.cross_dim = 128
        self.cross_heads = 4
        self.cross_q = nn.Linear(E, self.cross_dim, bias=False)
        self.cross_k = nn.Linear(F, self.cross_dim, bias=False)
        self.cross_v = nn.Linear(F, self.cross_dim, bias=False)
        self.cross_out = nn.Linear(E + self.cross_dim, E)
        self.cross_norm = nn.LayerNorm(E)

        # Stage 2: self-attention with relational bias
        self.rel_proj = nn.Linear(4, config.n_heads, bias=False)
        self.planet_blocks = nn.ModuleList([
            PlanetBlock(E, config.n_heads, config.ffn_hidden, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Stage 3: attention pooling + global MLP
        self.planet_pool_q = nn.Parameter(torch.empty(1, 1, E))
        self.fleet_pool_q = nn.Parameter(torch.empty(1, 1, F))
        nn.init.normal_(self.planet_pool_q, std=0.02)
        nn.init.normal_(self.fleet_pool_q, std=0.02)
        self.global_mlp = nn.Sequential(
            nn.Linear(E + F + Dg, G), nn.GELU(), nn.Linear(G, G), nn.LayerNorm(G)
        )

        # Stage 4: LSTM recurrence
        self.lstm = nn.LSTM(input_size=G, hidden_size=G, num_layers=1, batch_first=True)
        self.lstm_bypass_linear = nn.Linear(self.config.G, self.config.G)
        self.lstm_bypass_norm = nn.LayerNorm(self.config.G)

        # Action heads
        self.action_type_head = nn.Linear(E + G, 3)
        self.W_query = nn.Linear(E + G, E, bias=False)
        self.W_key = nn.Linear(E, E, bias=False)
        self._scale = math.sqrt(E)

        # Autoregressive conditioning embeddings
        _cond_dim = E + G  # dimension for conditioning vectors
        self.at_embedding = nn.Embedding(3, _cond_dim)   # 3 action types
        self.dist_embedding = nn.Linear(1, _cond_dim, bias=False)  # distance scalar → embedding

        # Amount head with autoregressive input: h_prime(E+G) ++ planet_ctx_tgt(E) ++ dist_emb(E+G)
        _amt_in = 2 * _cond_dim + E  # 2*(E+G) + E
        self.amount_head = nn.Sequential(
            nn.Linear(_amt_in, config.ffn_hidden),
            nn.GELU(),
            nn.Linear(config.ffn_hidden, config.n_amount_bins),
        )

        # Auxiliary prediction heads (global-level)
        self.aux_outcome_head = nn.Linear(G, 1)
        self.aux_return_10_head = nn.Linear(G, 1)
        self.aux_return_50_head = nn.Linear(G, 1)
        # Per-planet auxiliary heads
        self.aux_ownership_10_head = nn.Linear(E, 1)
        self.aux_opponent_launch_head = nn.Linear(E, 1)

        # Value heads
        self.v_outcome_head = nn.Sequential(
            nn.Linear(G, G // 2), nn.GELU(), nn.Linear(G // 2, 1), nn.Tanh()
        )
        self.v_score_diff_head = nn.Sequential(
            nn.Linear(G, G // 2), nn.GELU(), nn.Linear(G // 2, 1)
        )
        self.v_shaped_head = nn.Sequential(
            nn.Linear(G, G // 2), nn.GELU(), nn.Linear(G // 2, 1)
        )

    def forward(
        self,
        planet_features: torch.Tensor,
        fleet_features: torch.Tensor,
        fleet_mask: torch.Tensor,
        global_features: torch.Tensor,
        planet_mask: torch.Tensor,
        relational_tensor: Optional[torch.Tensor] = None,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple["PlanetPolicyOutput", Tuple[torch.Tensor, torch.Tensor]]:
        B, P, _ = planet_features.shape
        E, F, G = self.config.E, self.config.F, self.config.G
        device = planet_features.device

        # Guard: no real planets in the whole batch
        if not planet_mask.any():
            zeros_bp3 = torch.zeros(B, P, 3, device=device)
            zeros_bpp = torch.zeros(B, P, P, device=device)
            zeros_bpn = torch.zeros(B, P, self.config.n_amount_bins, device=device)
            zeros_b1  = torch.zeros(B, 1, device=device)
            dummy_h = torch.zeros(1, B, self.config.G, device=device)
            dummy_c = torch.zeros(1, B, self.config.G, device=device)
            return PlanetPolicyOutput(
                action_type_logits=zeros_bp3,
                target_logits=zeros_bpp,
                amount_logits=zeros_bpn,
                v_outcome=zeros_b1,
                v_score_diff=zeros_b1,
                v_shaped=zeros_b1,
            ), (dummy_h, dummy_c)

        # Stage 0: encode
        planet_emb = self.planet_encoder(planet_features)   # (B, P, E)
        fleet_emb  = self.fleet_encoder(fleet_features)     # (B, F_max, F)

        # Stage 1: cross-attention fleets→planets
        # Q from planets, K/V from fleets; dims differ so use manual matmul
        cd = self.cross_dim
        ch = self.cross_heads
        head_dim = cd // ch

        Q = self.cross_q(planet_emb)   # (B, P, cd)
        K = self.cross_k(fleet_emb)    # (B, F_max, cd)
        V = self.cross_v(fleet_emb)    # (B, F_max, cd)

        # Reshape to (B*ch, seq, head_dim)
        Q = Q.view(B, P, ch, head_dim).permute(0, 2, 1, 3).reshape(B * ch, P, head_dim)
        K = K.view(B, -1, ch, head_dim).permute(0, 2, 1, 3).reshape(B * ch, -1, head_dim)
        V = V.view(B, -1, ch, head_dim).permute(0, 2, 1, 3).reshape(B * ch, -1, head_dim)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(head_dim)  # (B*ch, P, F_max)

        # Fleet padding mask: mask out padding positions with -inf
        # fleet_mask: (B, F_max) True=real → we want True=keep, False=mask
        fleet_pad = ~fleet_mask  # (B, F_max) True=padding
        fleet_pad_expanded = fleet_pad.unsqueeze(1).expand(-1, ch, -1).reshape(B * ch, 1, -1)
        scores = scores.masked_fill(fleet_pad_expanded, float('-inf'))

        # Handle fully-masked fleet case (all padding): replace -inf rows with 0
        all_masked = fleet_pad_expanded.all(dim=-1, keepdim=True)  # (B*ch, P, 1)
        scores = scores.masked_fill(all_masked.expand_as(scores), 0.0)

        attn_weights = torch.softmax(scores, dim=-1)
        cross_out = torch.bmm(attn_weights, V)  # (B*ch, P, head_dim)
        cross_out = cross_out.reshape(B, ch, P, head_dim).permute(0, 2, 1, 3).reshape(B, P, cd)

        planet_emb = self.cross_norm(self.cross_out(torch.cat([planet_emb, cross_out], dim=-1)))

        # Stage 2: self-attention with relational bias
        rel_bias = None
        if relational_tensor is not None:
            rel_bias = self.rel_proj(relational_tensor)      # (B, P, P, n_heads)
            rel_bias = rel_bias.permute(0, 3, 1, 2)          # (B, n_heads, P, P)
            rel_bias = rel_bias.reshape(B * self.config.n_heads, P, P)

        key_pad = ~planet_mask  # (B, P) True=padding
        x = planet_emb
        for block in self.planet_blocks:
            x = block(x, attn_mask=rel_bias, key_padding_mask=key_pad)
        planet_ctx = x  # (B, P, E)

        # Stage 3: attention pooling
        scale_e = math.sqrt(E)
        scale_f = math.sqrt(F)

        # Planet pooling
        pq = self.planet_pool_q.expand(B, -1, -1)          # (B, 1, E)
        p_scores = torch.bmm(pq, planet_ctx.transpose(1, 2)) / scale_e  # (B, 1, P)
        p_pad = (~planet_mask).unsqueeze(1)                 # (B, 1, P)
        p_scores = p_scores.masked_fill(p_pad, float('-inf'))
        # Guard fully-masked
        p_all_masked = p_pad.all(dim=-1, keepdim=True)
        p_scores = p_scores.masked_fill(p_all_masked.expand_as(p_scores), 0.0)
        planet_pool = torch.bmm(torch.softmax(p_scores, dim=-1), planet_ctx).squeeze(1)  # (B, E)

        # Fleet pooling
        fq = self.fleet_pool_q.expand(B, -1, -1)           # (B, 1, F)
        f_scores = torch.bmm(fq, fleet_emb.transpose(1, 2)) / scale_f  # (B, 1, F_max)
        f_pad = (~fleet_mask).unsqueeze(1)                  # (B, 1, F_max)
        f_scores = f_scores.masked_fill(f_pad, float('-inf'))
        f_all_masked = f_pad.all(dim=-1, keepdim=True)
        f_scores = f_scores.masked_fill(f_all_masked.expand_as(f_scores), 0.0)
        fleet_pool = torch.bmm(torch.softmax(f_scores, dim=-1), fleet_emb).squeeze(1)  # (B, F)

        global_repr = self.global_mlp(
            torch.cat([planet_pool, fleet_pool, global_features], dim=-1)
        )  # (B, G)

        # Stage 4: LSTM recurrence
        if self.config.lstm_bypass:
            lstm_out = self.lstm_bypass_norm(self.lstm_bypass_linear(global_repr))
            B_dyn = global_repr.shape[0]
            _dev = global_repr.device
            new_hidden = (
                torch.zeros(1, B_dyn, self.config.G, device=_dev),
                torch.zeros(1, B_dyn, self.config.G, device=_dev),
            )
        else:
            lstm_out_seq, new_hidden = self.lstm(global_repr.unsqueeze(1), hidden_state)
            lstm_out = lstm_out_seq.squeeze(1)  # (B, G)

        # ── Autoregressive action head decode ──────────────────────────────────────

        # h_broadcast: (B, P, G) — lstm_out broadcast to per-planet
        h_broadcast = lstm_out.unsqueeze(1).expand(-1, P, -1)  # (B, P, G)

        # Concatenate planet ctx + lstm state
        planet_ctx_h = torch.cat([planet_ctx, h_broadcast], dim=-1)  # (B, P, E+G)

        # Step 1: action type head (no conditioning)
        action_type_logits = self.action_type_head(planet_ctx_h)  # (B, P, 3)

        # Step 2: target head — conditioned on action type
        at_idx = action_type_logits.argmax(dim=-1).clamp(0, 2)  # (B, P)
        at_emb = self.at_embedding(at_idx)                       # (B, P, E+G)
        h_prime = planet_ctx_h + at_emb                          # (B, P, E+G)  residual add

        # Pointer attention for target (using h_prime)
        query = self.W_query(h_prime)                            # (B, P, E)
        key   = self.W_key(planet_ctx)                           # (B, P, E)
        target_logits = torch.bmm(query, key.transpose(1, 2)) / self._scale  # (B, P, P)

        # Mask padding planets in target
        tgt_pad = (~planet_mask).unsqueeze(1).expand(-1, P, -1)
        target_logits = target_logits.masked_fill(tgt_pad, float('-inf'))

        # Step 3: amount head — conditioned on action type + target planet context
        tgt_idx = target_logits.argmax(dim=-1).clamp(0, P - 1)  # (B, P)
        # Gather planet context for the predicted target planet
        tgt_idx_exp = tgt_idx.unsqueeze(-1).expand(-1, -1, self.config.E)  # (B, P, E)
        planet_ctx_tgt = planet_ctx.gather(1, tgt_idx_exp)               # (B, P, E)
        # Distance embedding: use relational tensor distance feature if available
        if relational_tensor is not None:
            # relational_tensor: (B, P, P, 4), index [0] is distance_norm
            dist_vals = relational_tensor.gather(
                2, tgt_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, relational_tensor.shape[-1])
            ).squeeze(2)[:, :, 0:1]  # (B, P, 1)
        else:
            dist_vals = torch.zeros(B, P, 1, device=device)
        dist_emb = self.dist_embedding(dist_vals)  # (B, P, E+G)
        amount_in = torch.cat([h_prime, planet_ctx_tgt, dist_emb], dim=-1)  # (B, P, 2*(E+G)+E)
        amount_logits = self.amount_head(amount_in)  # (B, P, n_amount_bins)

        # Value heads
        v_outcome    = self.v_outcome_head(lstm_out)    # (B, 1)
        v_score_diff = self.v_score_diff_head(lstm_out) # (B, 1)
        v_shaped     = self.v_shaped_head(lstm_out)     # (B, 1)

        # Auxiliary heads
        aux_outcome = torch.tanh(self.aux_outcome_head(lstm_out))        # (B, 1)
        aux_return_10 = self.aux_return_10_head(lstm_out)                # (B, 1)
        aux_return_50 = self.aux_return_50_head(lstm_out)                # (B, 1)
        aux_ownership_10 = self.aux_ownership_10_head(planet_ctx)        # (B, P, 1)
        aux_opponent_launch = self.aux_opponent_launch_head(planet_ctx)  # (B, P, 1)
        aux_ownership_10 = aux_ownership_10.squeeze(-1)                  # (B, P)
        aux_opponent_launch = aux_opponent_launch.squeeze(-1)            # (B, P)

        return PlanetPolicyOutput(
            action_type_logits=action_type_logits,
            target_logits=target_logits,
            amount_logits=amount_logits,
            v_outcome=v_outcome,
            v_score_diff=v_score_diff,
            v_shaped=v_shaped,
            aux_outcome=aux_outcome,
            aux_return_10=aux_return_10,
            aux_return_50=aux_return_50,
            aux_ownership_10=aux_ownership_10,
            aux_opponent_launch=aux_opponent_launch,
        ), new_hidden
