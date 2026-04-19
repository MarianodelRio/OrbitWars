import math
import pytest
from dataclasses import dataclass

@dataclass
class FakePlanet:
    x: float
    y: float

from game.logic.geometry import dist, nearest_planet, fleet_speed, angle_to, eta, orbit_predict, path_crosses_sun

def test_dist_basic():
    a = FakePlanet(0, 0)
    b = FakePlanet(3, 4)
    assert dist(a, b) == pytest.approx(5.0)

def test_fleet_speed_one_ship():
    assert fleet_speed(1) == pytest.approx(1.0)

def test_fleet_speed_1000_ships():
    assert fleet_speed(1000) == pytest.approx(6.0)

def test_fleet_speed_intermediate():
    s = fleet_speed(500)
    assert 1.0 < s < 6.0

def test_angle_to():
    src = FakePlanet(0, 0)
    tgt = FakePlanet(1, 0)
    assert angle_to(src, tgt) == pytest.approx(0.0)
    tgt2 = FakePlanet(0, 1)
    assert angle_to(src, tgt2) == pytest.approx(math.pi / 2)

def test_eta_positive():
    src = FakePlanet(0, 0)
    tgt = FakePlanet(10, 0)
    assert eta(src, tgt, 1) > 0

def test_orbit_predict_no_steps():
    p = FakePlanet(60, 50)
    x, y = orbit_predict(p, 0.025, 0)
    assert x == pytest.approx(60.0)
    assert y == pytest.approx(50.0)

def test_orbit_predict_moves():
    p = FakePlanet(60, 50)
    x1, y1 = orbit_predict(p, 0.025, 10)
    assert not (x1 == pytest.approx(60.0) and y1 == pytest.approx(50.0))

def test_path_crosses_sun_direct_hit():
    assert path_crosses_sun(50, 30, 50, 70) is True

def test_path_crosses_sun_miss():
    assert path_crosses_sun(0, 0, 100, 0) is False

def test_path_crosses_sun_tangent():
    # horizontal line at y=40, x from 30 to 70 — misses sun (radius 10, center 50,50)
    assert path_crosses_sun(30, 40, 70, 40) is False

def test_nearest_planet():
    a = FakePlanet(0, 0)
    b = FakePlanet(10, 0)
    c = FakePlanet(3, 0)
    result = nearest_planet(a, [b, c])
    assert result is c
