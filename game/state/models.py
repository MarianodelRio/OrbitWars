from dataclasses import dataclass, field


@dataclass
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: float
    production: float


@dataclass
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: float


@dataclass
class GameState:
    step: int
    player: int
    angular_velocity: float
    planets: list
    fleets: list
    my_planets: list
    enemy_planets: list
    neutral_planets: list
    comet_planet_ids: list = field(default_factory=list)
    initial_planets: list = field(default_factory=list)


def parse_obs(obs) -> GameState:
    player = obs["player"]
    step = obs.get("step", 0)
    angular_velocity = obs.get("angular_velocity", 0.0)

    planets = []
    for p in obs.get("planets", []):
        planets.append(Planet(
            id=p[0], owner=p[1], x=p[2], y=p[3],
            radius=p[4], ships=p[5], production=p[6]
        ))

    fleets = []
    for f in obs.get("fleets", []):
        fleets.append(Fleet(
            id=f[0], owner=f[1], x=f[2], y=f[3],
            angle=f[4], from_planet_id=f[5], ships=f[6]
        ))

    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
    neutral_planets = [p for p in planets if p.owner == -1]

    comet_planet_ids = obs.get("comet_planet_ids", [])
    initial_planets_raw = obs.get("initial_planets", [])
    initial_planets = []
    for p in initial_planets_raw:
        initial_planets.append(Planet(id=p[0], owner=p[1], x=p[2], y=p[3], radius=p[4], ships=p[5], production=p[6]))

    return GameState(
        step=step,
        player=player,
        angular_velocity=angular_velocity,
        planets=planets,
        fleets=fleets,
        my_planets=my_planets,
        enemy_planets=enemy_planets,
        neutral_planets=neutral_planets,
        comet_planet_ids=comet_planet_ids,
        initial_planets=initial_planets,
    )
