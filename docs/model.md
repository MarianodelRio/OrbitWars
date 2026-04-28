# Model Architecture — PlanetPolicyModel

## Overview

`PlanetPolicyModel` is an entity-centric policy/value model for Orbit Wars. Each planet makes an independent decision (action type, target, amount), but all planets share context through cross-attention with fleets and self-attention across planets. A single LSTM step integrates global recurrent state across turns. Three value heads predict different return signals; auxiliary per-planet heads provide extra supervision.

## Config — PlanetPolicyConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `Dp` | int | 24 | Planet feature dimension |
| `Df` | int | 16 | Fleet feature dimension |
| `Dg` | int | 16 | Global feature dimension |
| `E` | int | 192 | Planet embedding size |
| `F` | int | 128 | Fleet embedding size |
| `G` | int | 384 | Global/LSTM hidden size |
| `n_heads` | int | 8 | Attention heads in self-attention blocks |
| `n_layers` | int | 4 | Number of PlanetBlock transformer layers |
| `ffn_hidden` | int | 768 | FFN hidden size inside each PlanetBlock |
| `dropout` | float | 0.1 | Dropout rate |
| `max_planets` | int | 50 | Maximum padded planet slots |
| `max_fleets` | int | 200 | Maximum padded fleet slots |
| `n_amount_bins` | int | 8 | Number of discrete ship-fraction bins |
| `lstm_bypass` | bool | False | If True, skip LSTM; use a linear projection instead |

## Architecture (5 Stages)

```
planet_features (B,P,24)    fleet_features (B,FL,16)    global_features (B,16)
        |                           |
  [Stage 0]                   [Stage 0]
  planet_encoder              fleet_encoder
  Linear→GELU→Linear→LN      Linear→GELU→Linear→LN
  (B,P,E)                     (B,FL,F)
        |                           |
        +-------[Stage 1]-----------+
              cross-attention (fleets→planets)
              Q from planets (E→cd), K/V from fleets (F→cd)
              4 heads, cd=128
              cross_out → cat → Linear → LN
              (B,P,E)
                    |
              [Stage 2]
              relational_bias (B,P,P,4) → rel_proj → (B,n_heads,P,P)
              4× PlanetBlock (pre-LN self-attn + FFN)
              planet_ctx (B,P,E)
                    |                   |
              [Stage 3]          attention-pool
              planet attention-pool     fleet attention-pool
              (B,E)                     (B,F)
                    \                  /          global_features(B,16)
                     cat → global_mlp(E+F+Dg→G→G→LN) → global_repr(B,G)
                                  |
                            [Stage 4]
                            LSTM (G→G, 1 layer)  [or bypass linear]
                            lstm_out (B,G)
                                  |
             +--------------------+-------------------+
             |                    |                   |
    action_type head       target (pointer)      amount head
    [E+G → 3]              [E+G → E → pointer]  [2*(E+G)+E → ffn_hidden → n_bins]
    (B,P,3)                (B,P,P)               (B,P,n_amount_bins)
             |
      value heads (3×): v_outcome, v_score_diff, v_shaped
      aux heads (5×):   aux_outcome, aux_return_10, aux_return_50,
                        aux_ownership_10, aux_opponent_launch
```

## Stage Detail

### Stage 0 — Encoders

Two separate MLPs encode raw features into fixed-size embeddings:

- **planet_encoder**: `Linear(Dp, E) → GELU → Linear(E, E) → LayerNorm(E)`
- **fleet_encoder**: `Linear(Df, F) → GELU → Linear(F, F) → LayerNorm(F)`

### Stage 1 — Cross-Attention (Fleets → Planets)

Manual multi-head cross-attention. Planets are queries (projected to `cd=128`), fleets are keys and values. Uses `ch=4` heads with `head_dim = cd // ch = 32`. Fleet padding positions are masked with `-inf` before softmax. Fully-padded rows (no real fleets) are set to zero. Output is concatenated with the planet embedding and projected: `Linear(E + cd, E)` followed by `LayerNorm(E)`.

### Stage 2 — Self-Attention with Relational Bias

The relational tensor `(B, P, P, 4)` is projected to `(B, n_heads, P, P)` and used as an additive attention-mask bias (passed as `attn_mask` to PyTorch `MultiheadAttention`). Four `PlanetBlock` layers apply pre-LN self-attention and pre-LN FFN. Padding planet positions are masked via `key_padding_mask`. Output is `planet_ctx (B, P, E)`.

### Stage 3 — Attention Pooling + Global MLP

Learned query vectors (`planet_pool_q`, `fleet_pool_q`) pool `planet_ctx` and `fleet_emb` into single vectors `(B, E)` and `(B, F)` using scaled dot-product softmax. These are concatenated with `global_features (B, Dg)` and passed through `global_mlp`: `Linear(E+F+Dg, G) → GELU → Linear(G, G) → LayerNorm(G)`.

### Stage 4 — LSTM Recurrence

`global_repr (B, G)` is fed as a single-step sequence into a 1-layer LSTM with `hidden_size=G`. The hidden state `(h, c)` carries episode context across turns. If `lstm_bypass=True`, a `Linear(G, G) + LayerNorm(G)` replaces the LSTM and dummy zero hidden states are returned.

**Autoregressive action decoding:**

1. `planet_ctx_h = cat[planet_ctx, lstm_out_broadcast] (B, P, E+G)`
2. Action type logits: `Linear(E+G, 3)` — no conditioning.
3. Target logits: argmax action type → `at_embedding(3, E+G)` adds residual → pointer attention `W_query(E+G→E)` dot `W_key(E→E)` → `(B, P, P)`.
4. Amount logits: gather target planet context, compute distance embedding from relational tensor channel 0, concatenate `[h_prime, planet_ctx_tgt, dist_emb] (2*(E+G)+E)` → `Linear → GELU → Linear → n_bins`.

## Output Heads

| Head | Module | Output Shape |
|---|---|---|
| `action_type_head` | `Linear(E+G, 3)` | `(B, P, 3)` |
| pointer target | `W_query(E+G,E)` · `W_key(E,E)` | `(B, P, P)` |
| `amount_head` | `Linear(2*(E+G)+E, ffn_hidden) → GELU → Linear(ffn_hidden, n_bins)` | `(B, P, n_amount_bins)` |
| `v_outcome_head` | `Linear(G, G//2) → GELU → Linear → Tanh` | `(B, 1)` |
| `v_score_diff_head` | `Linear(G, G//2) → GELU → Linear` | `(B, 1)` |
| `v_shaped_head` | `Linear(G, G//2) → GELU → Linear` | `(B, 1)` |
| `aux_outcome_head` | `Linear(G, 1)` + Tanh | `(B, 1)` |
| `aux_return_10_head` | `Linear(G, 1)` | `(B, 1)` |
| `aux_return_50_head` | `Linear(G, 1)` | `(B, 1)` |
| `aux_ownership_10_head` | `Linear(E, 1)` (per-planet) | `(B, P)` |
| `aux_opponent_launch_head` | `Linear(E, 1)` (per-planet) | `(B, P)` |

## Output Dataclass

`PlanetPolicyOutput` fields:

| Field | Shape | Notes |
|---|---|---|
| `action_type_logits` | `(B, P, 3)` | 0=NO_OP, 1=LAUNCH (third class unused) |
| `target_logits` | `(B, P, P)` | Pointer over planet slots; padding masked |
| `amount_logits` | `(B, P, n_amount_bins)` | Discrete ship-fraction bin logits |
| `v_outcome` | `(B, 1)` | Win/loss prediction, tanh-bounded |
| `v_score_diff` | `(B, 1)` | Score differential prediction, unbounded |
| `v_shaped` | `(B, 1)` | Shaped return prediction, unbounded |
| `aux_outcome` | `(B, 1)` | Auxiliary win/loss head (optional) |
| `aux_return_10` | `(B, 1)` | Auxiliary 10-step return (optional) |
| `aux_return_50` | `(B, 1)` | Auxiliary 50-step return (optional) |
| `aux_ownership_10` | `(B, P)` | Per-planet ownership prediction (optional) |
| `aux_opponent_launch` | `(B, P)` | Per-planet opponent launch prediction (optional) |

## Relational Tensor

Built by `StateBuilder._build_relational_tensor(planets, n_planets)`. Shape: `(max_planets, max_planets, 4)`.

| Channel | Feature | Formula |
|---|---|---|
| 0 | `distance_norm` | `min(euclidean_distance(i, j) / 100.0, 1.0)` |
| 1 | `angle_diff / pi` | `abs(atan2(yj-yi, xj-xi) - atan2(yi-50, xi-50)) / pi` (wrapped to `[-pi,pi]`) |
| 2 | `same_owner` | `1.0` if `owner[i] == owner[j]` else `0.0` |
| 3 | `is_reachable_in_50_turns` | `1.0` if `distance(i,j) <= 6.0 * 50.0` else `0.0` |

Diagonal entries (`i == j`) are left at zero.

## Planet Features (24 dims)

| Index | Feature | Normalization |
|---|---|---|
| 0 | is_mine | `float(owner == player)` |
| 1 | is_enemy | `float(owner not in (-1, player))` |
| 2 | is_neutral | `float(owner == -1)` |
| 3 | x_norm | `x / 100.0` |
| 4 | y_norm | `y / 100.0` |
| 5 | ships_log | `log(1 + ships) / log(1001)` |
| 6 | production_norm | `production / 5.0` |
| 7 | radius_norm | `radius / 3.0` |
| 8 | is_comet | `float(planet_id in comet_set)` |
| 9 | dist_from_sun | `hypot(x-50, y-50) / 50.0` |
| 10 | is_orbital | `1.0` if `dist_from_sun + radius < 50` and not comet |
| 11 | orbital_sin_theta | `sin(theta)` if orbital else `0.0` |
| 12 | orbital_cos_theta | `cos(theta)` if orbital else `0.0` |
| 13 | angular_velocity_norm | `angular_velocity / 0.05` if orbital else `0.0` |
| 14 | orbit_radius_norm | `dist_from_sun / 50.0` |
| 15 | enemy_threat_log | `log(1 + enemy_ship_log_sum) / log(1001)` |
| 16 | friendly_support_log | `log(1 + friendly_ship_log_sum) / log(1001)` |
| 17 | min_enemy_eta_norm | `min(min_enemy_eta, 50) / 50.0` (0 if no threat) |
| 18 | min_friendly_eta_norm | `min(min_friendly_eta, 50) / 50.0` (0 if none) |
| 19 | garrison_surplus | `clip((garrison - enemy_ships_incoming) / 200.0, -1, 1)` |
| 20 | nearest_enemy_dist | `min(dist_to_nearest_enemy, 100) / 100.0` |
| 21 | nearest_friendly_dist | `min(dist_to_nearest_friendly, 100) / 100.0` |
| 22 | n_enemy_within_30 | `min(count_enemy_planets_within_30, 10) / 10.0` |
| 23 | is_frontline | `1.0` if `nearest_enemy_dist < 30.0` else `0.0` |

Fleet threat features (15–18) count only fleets whose heading is within `pi/4` of the planet direction.

## Fleet Features (16 dims)

| Index | Feature | Normalization |
|---|---|---|
| 0 | is_mine | `float(owner == player)` |
| 1 | is_enemy | `float(owner != player)` |
| 2 | x_norm | `x / 100.0` |
| 3 | y_norm | `y / 100.0` |
| 4 | sin_angle | `sin(angle)` |
| 5 | cos_angle | `cos(angle)` |
| 6 | ships_log | `log(1 + ships) / log(1001)` |
| 7 | dist_from_sun | `hypot(x-50, y-50) / 50.0` |
| 8 | speed_norm | `min(speed / 6.0, 1.0)` where `speed = 1 + 5*(log(max(ships,1))/log(1000))^1.5` |
| 9 | best_target_dist | `min(dist_to_nearest_aligned_planet / 100.0, 1.0)` |
| 10 | best_target_eta | `min(dist / max(speed,1e-6) / 50.0, 1.0)` |
| 11 | target_is_mine | `1.0` if nearest-aligned planet owned by player |
| 12 | target_is_enemy | `1.0` if nearest-aligned planet owned by another player |
| 13 | reserved | `0.0` |
| 14 | reserved | `0.0` |
| 15 | reserved | `0.0` |

"Nearest aligned planet" is the closest planet whose direction from the fleet is within `pi/4` of the fleet heading.

## Global Features (16 dims)

| Index | Feature | Normalization |
|---|---|---|
| 0 | turn_norm | `turn / 500.0` |
| 1 | my_ship_share | `my_ships / total_ships` (0.5 if total=0) |
| 2 | my_planet_share | `my_planets / total_planets` |
| 3 | fleet_density | `min(n_fleets, max_fleets) / 200.0` |
| 4 | centered_turn | `(turn - 250) / 250.0` |
| 5 | enemy_max_ship_share | max over enemies of `enemy_ships / total_ships` |
| 6 | neutral_share | `neutral_ships / total_ships` |
| 7 | my_production_share | `my_production / total_production` |
| 8 | enemy_max_planet_share | max over enemies of `enemy_planet_count / total_planets` |
| 9 | n_my_fleets_norm | `min(n_my_fleets / 200.0, 1.0)` |
| 10 | n_enemy_fleets_norm | `min(n_enemy_fleets / 200.0, 1.0)` |
| 11 | next_comet_spawn_norm | `max(turns_until_next_comet_spawn, 0) / 500.0` |
| 12 | phase_early | `1.0` if `turn < 100` |
| 13 | phase_mid | `1.0` if `100 <= turn < 350` |
| 14 | phase_late | `1.0` if `turn >= 350` |
| 15 | my_eliminated | `1.0` if player has zero planets and zero fleets |

## Action Codec

### BINS

`BINS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0]` — 8 discrete ship-fraction values (indices 0–7).

### Encode Path (`encode_per_planet`)

For each owned planet:
- If no action recorded: label `action_type = NO_OP (0)`, target and amount stay at `-1`.
- If an action is recorded: label `action_type = LAUNCH (1)`.
  - **Target inference**: iterate over all other planets; find the one whose direction from the source is closest (by angular difference) to the recorded fleet angle. If the best angular difference exceeds `angular_diff_threshold` (default `pi/4`), target index is set to `-1` (invalid).
  - **Amount bin**: compute `fraction = ships_launched / garrison`. Quantize to nearest bin in `BINS` using `argmin(abs(BINS - fraction))`.

### Decode Path (`decode_per_planet`)

Autoregressive — uses model's own argmax predictions at each step:
1. For each owned planet: skip if `argmax(action_type_logits) == NO_OP`.
2. Target: `argmax(target_logits[i])` over valid planets (self and same-id masked with `-inf`).
3. Amount: `argmax(amount_logits[i])`, clipped to `[0, len(BINS)-1]`.
4. Compute `n_ships = BINS[bin] * raw_ship_count`. Skip if `n_ships < 1.0`.
5. Compute launch angle from source to (optionally adjusted) target position.

### Orbital Position Adjustment

When `angular_velocity != 0`, the decoder estimates flight time (`eta_turns = distance / 1.0`) and rotates the target position by `eta_turns * angular_velocity` radians around the board center before computing the launch angle.

## LSTM Hidden State

`NeuralBot` stores the LSTM state as `self._hidden = (h, c)` where each has shape `(1, 1, G)` (1 layer, batch=1 at inference).

- On the first call after construction or `reset()`, `_hidden` is `None` and the LSTM uses zero initial state.
- After each `act()` call, `_hidden` is updated to the new `(h, c)` returned by the model, persisting across turns within an episode.
- `reset()` sets `_hidden = None`, discarding episode state. This must be called between episodes.
- If `lstm_bypass=True` in config, the LSTM is not executed; the model returns zero tensors for `(h, c)` every call, so `_hidden` carries no information between turns.
