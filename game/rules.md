# Orbit Wars — Game Rules

## Overview

Conquer planets rotating around a sun in continuous 2D space. A real-time strategy game for 2 or 4 players.

Players start with a single home planet and compete to control the map by sending fleets to capture neutral and enemy planets. The board is a 100×100 continuous space with a sun at the center. Planets orbit the sun, comets fly through on elliptical trajectories, and fleets travel in straight lines. The game lasts 500 turns. The player with the most total ships (on planets + in fleets) at the end wins.

---

## Board Layout

- **Board**: 100×100 continuous space, origin at top-left.
- **Sun**: Centered at (50, 50) with radius 10. Fleets that cross the sun are destroyed.
- **Symmetry**: All planets and comets are placed with 4-fold mirror symmetry around the center: (x, y), (100-x, y), (x, 100-y), (100-x, 100-y). This ensures fairness regardless of starting position.

---

## Planets

Each planet is represented as `[id, owner, x, y, radius, ships, production]`.

| Field | Description |
|---|---|
| `owner` | Player ID (0–3), or -1 for neutral |
| `radius` | Determined by production: `1 + ln(production)` |
| `production` | Integer 1–5. Owned planets generate this many ships per turn |
| `ships` | Current garrison. Starts between 5 and 99 (skewed low) |

### Planet Types

- **Orbiting planets**: Planets whose `orbital_radius + planet_radius < 50` rotate around the sun at a constant angular velocity (0.025–0.05 rad/turn, randomized per game). Use `initial_planets` and `angular_velocity` from the observation to predict their positions.
- **Static planets**: Planets further from the center do not rotate.

The map contains 20–40 planets (5–10 symmetric groups of 4). At least 3 groups are static and at least one group is orbiting.

### Home Planets

One symmetric group is randomly chosen as starting planets. In a 2-player game, players start on diagonally opposite planets (Q1 and Q4). In a 4-player game, each player gets one planet from the group. Home planets start with 10 ships.

---

## Fleets

Each fleet is represented as `[id, owner, x, y, angle, from_planet_id, ships]`.

| Field | Description |
|---|---|
| `angle` | Direction of travel in radians |
| `ships` | Number of ships (does not change during travel) |

### Fleet Speed

Speed scales logarithmically with size:

```
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
```

- 1 ship → speed 1.0
- ~500 ships → speed ~5.0
- 1000 ships → speed 6.0 (max)

### Fleet Movement

Fleets travel in a straight line at their computed speed each turn. A fleet is destroyed if it:
- Goes out of bounds (leaves the 100×100 field)
- Crosses the sun (path segment comes within sun's radius)
- Collides with any planet (triggers combat)

Collision detection is **continuous** — the entire path segment is checked, not just the endpoint.

### Fleet Launch

Each turn, return a list of moves: `[from_planet_id, direction_angle, num_ships]`.

- You can only launch from planets you own.
- You cannot launch more ships than the planet currently has.
- The fleet spawns just outside the planet's radius in the given direction.
- You can issue multiple launches from the same or different planets in a single turn.

---

## Comets

Comets are temporary objects that fly through on highly elliptical orbits. They spawn in groups of 4 (one per quadrant) at steps **50, 150, 250, 350, and 450**.

| Property | Value |
|---|---|
| Radius | 1.0 (fixed) |
| Production | 1 ship/turn when owned |
| Starting ships | Random, skewed low (minimum of 4 rolls from 1–99) |
| Speed | 4.0 units/turn (default) |

- Comets appear in `planets` and follow all normal planet rules (capture, production, fleet launch, combat).
- Check `comet_planet_ids` in the observation to identify them.
- When a comet leaves the board, it is removed **along with all ships garrisoned on it**.
- Comets are removed **before** fleet launches each turn — you cannot launch from a departing comet.
- The `comets` field contains full trajectory paths for predicting future positions.

---

## Turn Order

Each turn executes in this order:

1. **Comet expiration**: Remove comets that have left the board.
2. **Comet spawning**: Spawn new comet groups at designated steps.
3. **Fleet launch**: Process all player actions, creating new fleets.
4. **Production**: All owned planets (including comets) generate ships.
5. **Fleet movement**: Move all fleets. Check for out-of-bounds, sun collision, and planet collision. Colliding fleets are queued for combat.
6. **Planet rotation & comet movement**: Orbiting planets rotate, comets advance. Fleets swept by a moving planet/comet are pulled into combat.
7. **Combat resolution**: Resolve all queued planet combats.

---

## Combat

When one or more fleets collide with a planet:

1. All arriving fleets are grouped by owner. Same-owner ships are summed.
2. The largest attacking force fights the second largest — the **difference** survives.
3. If there is a surviving attacker:
   - **Same owner as planet** → surviving ships are added to the garrison.
   - **Different owner** → surviving ships fight the garrison. If attackers exceed the garrison, the planet changes ownership and the garrison becomes the surplus.
4. **Tie between attackers** → all attacking ships are destroyed.

---

## Scoring and Termination

The game ends when:
- **Step limit**: 500 turns elapsed.
- **Elimination**: Only one player (or zero) remains with any planets or fleets.

**Final score** = total ships on owned planets + total ships in owned fleets. Highest score wins.

---

## Observation Reference

| Field | Type | Description |
|---|---|---|
| `planets` | `[[id, owner, x, y, radius, ships, production], ...]` | All planets including comets |
| `fleets` | `[[id, owner, x, y, angle, from_planet_id, ships], ...]` | All active fleets |
| `player` | `int` | Your player ID (0–3) |
| `angular_velocity` | `float` | Planet rotation speed (radians/turn) |
| `initial_planets` | `[[id, owner, x, y, radius, ships, production], ...]` | Planet positions at game start |
| `comets` | `[{planet_ids, paths, path_index}, ...]` | Active comet group data |
| `comet_planet_ids` | `[int, ...]` | Planet IDs that are comets |
| `remainingOverageTime` | `float` | Remaining overage time budget (seconds) |

---

## Action Format

Return a list of moves:

```python
[[from_planet_id, direction_angle, num_ships], ...]
```

- `from_planet_id`: ID of a planet you own.
- `direction_angle`: Angle in radians (0 = right, π/2 = down).
- `num_ships`: Integer number of ships to send.

Return `[]` to take no action.

---

## Agent Convenience

```python
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT

def agent(obs):
    planets = [Planet(*p) for p in obs.get("planets", [])]
    fleets = [Fleet(*f) for f in obs.get("fleets", [])]
    player = obs.get("player", 0)
    return []  # list of [from_planet_id, angle, num_ships]
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `episodeSteps` | 500 | Maximum number of turns |
| `actTimeout` | 1 | Seconds per turn |
| `shipSpeed` | 6.0 | Maximum fleet speed |
| `sunRadius` | 10.0 | Radius of the sun |
| `boardSize` | 100.0 | Board dimensions |
| `cometSpeed` | 4.0 | Comet speed (units/turn) |
