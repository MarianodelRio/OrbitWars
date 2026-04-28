# Game Mechanics

## Overview

Orbit Wars is a 2-player (or 4-player) real-time strategy game played on a 100x100 continuous board with a sun at the center. Players start with one home planet and compete to control the map by sending fleets. The game lasts 500 turns. The player with the most total ships at the end wins.

## Game Rules Reference

The canonical rules document is [`game/rules.md`](../game/rules.md). Everything below is a summary for quick reference; consult the canonical document for exact definitions.

## Observation Structure

Each call to `agent_fn` receives an `obs` dict with the following fields:

| Field | Type | Description |
|---|---|---|
| `planets` | list of `[id, owner, x, y, radius, ships, production]` | All planets including comets |
| `fleets` | list of `[id, owner, x, y, angle, from_planet_id, ships]` | All active fleets |
| `player` | int | Your player ID (0–3) |
| `step` | int | Current turn number (0-based) |
| `angular_velocity` | float | Orbital planet rotation speed (radians/turn) |
| `initial_planets` | list of `[id, owner, x, y, radius, ships, production]` | Planet positions at game start |
| `comets` | list of dicts | Active comet group data with `planet_ids`, `paths`, `path_index` |
| `comet_planet_ids` | list of int | Planet IDs that are currently comets |
| `remainingOverageTime` | float | Remaining overage time budget (seconds) |

Planet vector column indices:
- 0: id, 1: owner (-1=neutral, 0–3=player), 2: x, 3: y, 4: radius, 5: ships, 6: production

Fleet vector column indices:
- 0: id, 1: owner, 2: x, 3: y, 4: angle (radians), 5: from_planet_id, 6: ships

## Accessing State from Python

Minimal `agent_fn` skeleton:

```python
def agent_fn(obs, config=None):
    planets = obs.get("planets", [])
    fleets  = obs.get("fleets", [])
    player  = obs.get("player", 0)

    my_planets  = [p for p in planets if p[1] == player]
    enemy_planets = [p for p in planets if p[1] >= 0 and p[1] != player]
    neutral_planets = [p for p in planets if p[1] == -1]
    comet_ids = set(obs.get("comet_planet_ids", []))

    moves = []
    for source in my_planets:
        # [from_planet_id, direction_angle_radians, num_ships]
        # moves.append([source[0], angle, n_ships])
        pass
    return moves
```

## Turn Order

Each turn executes in this exact order:

1. **Comet expiration**: Remove comets that have left the board (ships on them are destroyed).
2. **Comet spawning**: Spawn new comet groups at steps 50, 150, 250, 350, and 450.
3. **Fleet launch**: Process all player actions — create new fleets from owned planets.
4. **Production**: All owned planets (including comets) generate ships equal to their production value.
5. **Fleet movement**: Move all fleets. Check for out-of-bounds, sun collision (fleet destroyed), and planet collision (fleet queued for combat). Collision detection is continuous — the entire path segment is checked.
6. **Planet rotation and comet movement**: Orbiting planets rotate; comets advance. Fleets swept by a moving planet or comet are pulled into combat.
7. **Combat resolution**: Resolve all queued planet combats.

## Combat Rules

When one or more fleets collide with a planet:

1. All arriving fleets are grouped by owner. Same-owner ships are summed.
2. The largest attacking group fights the second largest — the **difference** survives.
3. If there is a surviving attacker:
   - Same owner as planet garrison: surviving ships are **added** to the garrison.
   - Different owner: surviving ships fight the garrison. If attackers exceed the garrison, the planet changes ownership and the surplus becomes the new garrison.
4. Tie between attacking groups: all attacking ships are destroyed.

## Fleet Speed Formula

```
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
```

With `maxSpeed = 6.0`:
- 1 ship → speed 1.0
- ~500 ships → speed ~5.0
- 1000 ships → speed 6.0 (max)

This is the formula used by `StateBuilder` and `OracleSniperBot` when estimating ETA.

## Orbiting Planets

Planets whose `orbital_radius + planet_radius < 50` orbit the sun at a constant `angular_velocity` (0.025–0.05 rad/turn, randomized per game). Their positions at turn `t` can be predicted from `initial_planets` and `angular_velocity`:

```python
import math
theta_0 = math.atan2(init_y - 50, init_x - 50)
theta_t = theta_0 + t * angular_velocity
r = math.hypot(init_x - 50, init_y - 50)
predicted_x = 50 + r * math.cos(theta_t)
predicted_y = 50 + r * math.sin(theta_t)
```

When launching at an orbiting planet, aim for its predicted position at the expected arrival time, not its current position.

## Comets

Comets are temporary planets on highly elliptical orbits. They spawn in groups of 4 (one per quadrant) at steps 50, 150, 250, 350, and 450. Properties:

- Radius: 1.0 (fixed)
- Production: 1 ship/turn when owned
- Starting ships: random, skewed low (minimum of 4 rolls from 1–99)
- Speed: 4.0 units/turn

Comets obey all normal planet rules (combat, production, launch). When a comet leaves the board, it is removed along with all ships on it. Comets are removed **before** fleet launches each turn — you cannot launch from a departing comet. Use `comet_planet_ids` to identify comets in the `planets` list.

## Edge Cases

- You cannot launch more ships than the planet currently holds (the environment silently clamps).
- Launching 0 ships is a no-op.
- You can issue multiple launches from the same planet in a single turn (each action is processed independently).
- Fleets that cross the sun boundary `(dist_from_center <= 10)` are destroyed.
- Fleets that leave the 100x100 board are destroyed.
- In a 2-player game, players start on diagonally opposite planets (Q1 and Q4) for fairness.
- All planets and comets are placed with 4-fold mirror symmetry.
- At the step limit, the player with the most total ships (planets + fleets) wins. Ties are possible.
