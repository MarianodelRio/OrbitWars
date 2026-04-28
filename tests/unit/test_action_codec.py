"""Unit tests for ActionCodec per-planet encode/decode."""

import math

import numpy as np
import pytest

from bots.neural.action_codec import ActionCodec
from bots.neural.types import ActionContext


MAX_PLANETS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(n_planets=5, player=0):
    """Planets on a horizontal line: 0=(0,50), 1=(10,50), 2=(20,50) ..."""
    planet_ids = np.arange(n_planets, dtype=np.int32)
    positions = np.array([[i * 10.0, 50.0] for i in range(n_planets)], dtype=np.float32)
    n_mine = n_planets // 2 + 1
    owners = np.array([player] * n_mine + [1 - player] * (n_planets - n_mine), dtype=np.int32)
    my_mask = owners == player
    return ActionContext(
        planet_ids=planet_ids,
        planet_positions=positions,
        my_planet_mask=my_mask,
        n_planets=n_planets,
    )


def _make_planets(n=5, ships=100.0):
    """Planet array shape (n, 7); col 5 = ships."""
    arr = np.zeros((n, 7), dtype=np.float32)
    arr[:, 5] = ships
    return arr


# ---------------------------------------------------------------------------
# encode_per_planet — shape & dtype
# ---------------------------------------------------------------------------

def test_encode_output_shapes():
    codec = ActionCodec()
    ctx = _make_context(5)
    planets = _make_planets(5)
    labels = codec.encode_per_planet(np.empty((0, 3), dtype=np.float32), ctx, planets, 1.0, MAX_PLANETS)

    assert labels.planet_action_types.shape == (MAX_PLANETS,)
    assert labels.planet_target_idxs.shape == (MAX_PLANETS,)
    assert labels.planet_amount_bins.shape == (MAX_PLANETS,)
    assert labels.my_planet_mask.shape == (MAX_PLANETS,)


def test_encode_output_dtypes():
    codec = ActionCodec()
    ctx = _make_context(5)
    planets = _make_planets(5)
    labels = codec.encode_per_planet(np.empty((0, 3), dtype=np.float32), ctx, planets, 0.0, MAX_PLANETS)

    assert labels.planet_action_types.dtype == np.int32
    assert labels.planet_target_idxs.dtype == np.int32
    assert labels.planet_amount_bins.dtype == np.int32
    assert labels.my_planet_mask.dtype == bool
    assert isinstance(labels.value_target, float)


# ---------------------------------------------------------------------------
# encode_per_planet — no actions
# ---------------------------------------------------------------------------

def test_no_actions_my_planets_are_noop():
    codec = ActionCodec()
    ctx = _make_context(5)
    planets = _make_planets(5)
    labels = codec.encode_per_planet(np.empty((0, 3), dtype=np.float32), ctx, planets, 1.0, MAX_PLANETS)

    for i in range(ctx.n_planets):
        if ctx.my_planet_mask[i]:
            assert labels.planet_action_types[i] == ActionCodec.NO_OP, f"planet {i} should be NO_OP"


def test_no_actions_non_mine_are_padding():
    codec = ActionCodec()
    ctx = _make_context(5)
    planets = _make_planets(5)
    labels = codec.encode_per_planet(np.empty((0, 3), dtype=np.float32), ctx, planets, 1.0, MAX_PLANETS)

    for i in range(ctx.n_planets):
        if not ctx.my_planet_mask[i]:
            assert labels.planet_action_types[i] == -1
            assert labels.planet_target_idxs[i] == -1
            assert labels.planet_amount_bins[i] == -1


def test_padding_slots_beyond_n_planets_are_minus_one():
    codec = ActionCodec()
    ctx = _make_context(3)
    planets = _make_planets(3)
    labels = codec.encode_per_planet(np.empty((0, 3), dtype=np.float32), ctx, planets, 0.0, MAX_PLANETS)

    for i in range(3, MAX_PLANETS):
        assert labels.planet_action_types[i] == -1
        assert labels.planet_target_idxs[i] == -1
        assert labels.planet_amount_bins[i] == -1
        assert not labels.my_planet_mask[i]


# ---------------------------------------------------------------------------
# encode_per_planet — with a launch action
# ---------------------------------------------------------------------------

def test_launch_action_produces_launch_label():
    """An action from my planet should produce action_type=LAUNCH."""
    codec = ActionCodec(angular_diff_threshold=math.pi)  # accept any angle
    ctx = _make_context(5)
    planets = _make_planets(5, ships=100.0)

    # Source = planet 0 (position (0,50)), angle pointing right → planet 1 (10,50)
    source_id = int(ctx.planet_ids[0])
    raw_actions = np.array([[source_id, 0.0, 50.0]], dtype=np.float32)

    labels = codec.encode_per_planet(raw_actions, ctx, planets, 1.0, MAX_PLANETS)

    assert labels.planet_action_types[0] == ActionCodec.LAUNCH


def test_launch_action_target_idx_in_range():
    codec = ActionCodec(angular_diff_threshold=math.pi)
    ctx = _make_context(5)
    planets = _make_planets(5, ships=100.0)

    source_id = int(ctx.planet_ids[0])
    raw_actions = np.array([[source_id, 0.0, 50.0]], dtype=np.float32)
    labels = codec.encode_per_planet(raw_actions, ctx, planets, 1.0, MAX_PLANETS)

    # target_idx should be a valid planet index (not -1 with large threshold)
    if labels.planet_target_idxs[0] != -1:
        assert 0 <= labels.planet_target_idxs[0] < ctx.n_planets


def test_launch_action_amount_bin_in_range():
    codec = ActionCodec(angular_diff_threshold=math.pi)
    ctx = _make_context(5)
    planets = _make_planets(5, ships=100.0)

    source_id = int(ctx.planet_ids[0])
    raw_actions = np.array([[source_id, 0.0, 50.0]], dtype=np.float32)
    labels = codec.encode_per_planet(raw_actions, ctx, planets, 1.0, MAX_PLANETS)

    assert labels.planet_action_types[0] == ActionCodec.LAUNCH
    assert 0 <= labels.planet_amount_bins[0] <= 7


def test_ambiguous_target_marked_minus_one():
    """With threshold=0, an angle that doesn't match any planet → target_idx=-1."""
    codec = ActionCodec(angular_diff_threshold=0.0)
    ctx = _make_context(5)
    planets = _make_planets(5, ships=100.0)

    # angle = π/2 (pointing straight up): no planet lies directly above planet 0,
    # so all angular diffs are > 0 and the strict threshold rejects them.
    source_id = int(ctx.planet_ids[0])
    raw_actions = np.array([[source_id, math.pi / 2, 50.0]], dtype=np.float32)
    labels = codec.encode_per_planet(raw_actions, ctx, planets, 1.0, MAX_PLANETS)

    # action_type is still LAUNCH (has an action) but target_idx is -1
    assert labels.planet_action_types[0] == ActionCodec.LAUNCH
    assert labels.planet_target_idxs[0] == -1


def test_amount_bin_half_ships():
    """Sending half of 100 ships → fraction=0.5 → BINS=[.1,.25,.5,.75,1.0] → bin 2."""
    codec = ActionCodec(angular_diff_threshold=math.pi)
    ctx = _make_context(5)
    planets = _make_planets(5, ships=100.0)

    source_id = int(ctx.planet_ids[0])
    raw_actions = np.array([[source_id, 0.0, 50.0]], dtype=np.float32)
    labels = codec.encode_per_planet(raw_actions, ctx, planets, 1.0, MAX_PLANETS)

    assert labels.planet_action_types[0] == ActionCodec.LAUNCH
    assert labels.planet_amount_bins[0] == 4  # closest bin to 0.5


def test_my_planet_mask_matches_context():
    codec = ActionCodec()
    ctx = _make_context(6)
    planets = _make_planets(6)
    labels = codec.encode_per_planet(np.empty((0, 3), dtype=np.float32), ctx, planets, 0.0, MAX_PLANETS)

    for i in range(ctx.n_planets):
        assert labels.my_planet_mask[i] == ctx.my_planet_mask[i]


def test_value_target_preserved():
    codec = ActionCodec()
    ctx = _make_context(4)
    planets = _make_planets(4)
    labels = codec.encode_per_planet(np.empty((0, 3), dtype=np.float32), ctx, planets, -1.0, MAX_PLANETS)
    assert labels.value_target == -1.0


def test_multiple_actions_each_encoded():
    """Two launch actions from two different my-planets should both be LAUNCH."""
    codec = ActionCodec(angular_diff_threshold=math.pi)
    ctx = _make_context(6)
    planets = _make_planets(6, ships=100.0)

    # My planets are 0..3 (n_mine = 4 for n_planets=6 with n//2+1=4)
    src0 = int(ctx.planet_ids[0])
    src1 = int(ctx.planet_ids[1])
    raw_actions = np.array([
        [src0, 0.0, 50.0],
        [src1, 0.0, 30.0],
    ], dtype=np.float32)

    labels = codec.encode_per_planet(raw_actions, ctx, planets, 1.0, MAX_PLANETS)

    assert labels.planet_action_types[0] == ActionCodec.LAUNCH
    assert labels.planet_action_types[1] == ActionCodec.LAUNCH


# ---------------------------------------------------------------------------
# decode_per_planet
# ---------------------------------------------------------------------------

class _FakeOutput:
    """Minimal object mimicking PlanetPolicyOutput (numpy, not tensor)."""
    def __init__(self, action_type_logits, target_logits, amount_logits):
        self.action_type_logits = action_type_logits
        self.target_logits = target_logits
        self.amount_logits = amount_logits


def _make_output(n_planets=MAX_PLANETS, all_noop=False, launch_planet=0, target_planet=2, amount_bin=2):
    at = np.full((n_planets, 2), -100.0, dtype=np.float32)
    if all_noop:
        at[:, 0] = 100.0
    else:
        at[:, 0] = 100.0
        at[launch_planet, 0] = -100.0
        at[launch_planet, 1] = 100.0

    tgt = np.zeros((n_planets, n_planets), dtype=np.float32)
    tgt[launch_planet, target_planet] = 100.0

    amt = np.zeros((n_planets, 9), dtype=np.float32)
    amt[launch_planet, amount_bin] = 100.0

    return _FakeOutput(at, tgt, amt)


def test_decode_returns_list():
    codec = ActionCodec()
    ctx = _make_context(5)
    output = _make_output(all_noop=True)
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)
    planets[:, 5] = 0.5  # 100 ships each

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS)
    assert isinstance(result, list)


def test_decode_all_noop_returns_empty():
    codec = ActionCodec()
    ctx = _make_context(5)
    output = _make_output(all_noop=True)
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)
    planets[:, 5] = 0.5

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS)
    assert result == []


def test_decode_launch_returns_one_action():
    codec = ActionCodec()
    ctx = _make_context(5)
    # Planet 0 (my planet) launches toward planet 2
    output = _make_output(launch_planet=0, target_planet=2, amount_bin=4)
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)
    raw_ship_counts = np.zeros(MAX_PLANETS, dtype=np.float32)
    raw_ship_counts[0] = 100.0

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS, raw_ship_counts=raw_ship_counts)
    assert len(result) == 1


def test_decode_action_format_is_three_element():
    codec = ActionCodec()
    ctx = _make_context(5)
    output = _make_output(launch_planet=0, target_planet=2, amount_bin=4)
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)
    raw_ship_counts = np.zeros(MAX_PLANETS, dtype=np.float32)
    raw_ship_counts[0] = 100.0

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS, raw_ship_counts=raw_ship_counts)
    action = result[0]
    assert len(action) == 3
    planet_id, angle, n_ships = action
    assert isinstance(n_ships, float)
    assert n_ships > 0


def test_decode_planet_id_matches_source():
    codec = ActionCodec()
    ctx = _make_context(5)
    output = _make_output(launch_planet=0, target_planet=2, amount_bin=4)
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)
    raw_ship_counts = np.zeros(MAX_PLANETS, dtype=np.float32)
    raw_ship_counts[0] = 100.0

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS, raw_ship_counts=raw_ship_counts)
    planet_id = result[0][0]
    assert planet_id == int(ctx.planet_ids[0])


def test_decode_angle_correct_for_horizontal():
    """Planet 0 at (0,50) → planet 2 at (20,50): angle should be 0."""
    codec = ActionCodec()
    ctx = _make_context(5)
    output = _make_output(launch_planet=0, target_planet=2, amount_bin=4)
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)
    raw_ship_counts = np.zeros(MAX_PLANETS, dtype=np.float32)
    raw_ship_counts[0] = 100.0

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS, raw_ship_counts=raw_ship_counts)
    _, angle, _ = result[0]
    assert abs(angle) < 1e-4


def test_decode_skips_non_my_planets():
    """A LAUNCH decision on an enemy planet slot should be ignored."""
    codec = ActionCodec()
    n_real = 5
    ctx = _make_context(n_real)

    # Find first enemy planet
    enemy_idx = next(i for i in range(n_real) if not ctx.my_planet_mask[i])

    at = np.full((MAX_PLANETS, 2), -100.0, dtype=np.float32)
    at[enemy_idx, 1] = 100.0  # "LAUNCH" from an enemy planet — should be ignored
    tgt = np.zeros((MAX_PLANETS, MAX_PLANETS), dtype=np.float32)
    tgt[enemy_idx, 0] = 100.0
    amt = np.zeros((MAX_PLANETS, 5), dtype=np.float32)
    amt[enemy_idx, 4] = 100.0

    output = _FakeOutput(at, tgt, amt)
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)
    planets[:, 5] = 0.5

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS)
    assert result == []


def test_decode_zero_ships_skipped():
    """If source has 0 ships, action should be skipped (n_ships < 1)."""
    codec = ActionCodec()
    ctx = _make_context(5)
    output = _make_output(launch_planet=0, target_planet=2, amount_bin=0)
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)
    planets[0, 5] = 0.0  # 0 ships

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS)
    assert result == []


def test_decode_empty_context():
    codec = ActionCodec()
    ctx = ActionContext(
        planet_ids=np.empty(0, dtype=np.int32),
        planet_positions=np.empty((0, 2), dtype=np.float32),
        my_planet_mask=np.empty(0, dtype=bool),
        n_planets=0,
    )
    output = _make_output()
    planets = np.zeros((MAX_PLANETS, 10), dtype=np.float32)

    result = codec.decode_per_planet(output, ctx, planets, MAX_PLANETS)
    assert result == []
