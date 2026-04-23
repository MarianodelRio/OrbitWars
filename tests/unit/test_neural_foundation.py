"""Unit tests for the neural bot foundation: types, state_builder, action_codec."""

import math

import numpy as np
import pytest

from bots.neural.types import ActionContext, ModelInput, ModelLabels
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec
from bots.neural.model import PolicyOutput
from dataset.episode import StepRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(planets_rows, fleets_rows, actions_p0_rows=None, actions_p1_rows=None):
    """Build a minimal StepRecord from plain lists."""
    if len(planets_rows) == 0:
        planets = np.empty((0, 7), dtype=np.float32)
    else:
        planets = np.array(planets_rows, dtype=np.float32)

    if len(fleets_rows) == 0:
        fleets = np.empty((0, 7), dtype=np.float32)
    else:
        fleets = np.array(fleets_rows, dtype=np.float32)

    if actions_p0_rows is None or len(actions_p0_rows) == 0:
        actions_p0 = np.empty((0, 3), dtype=np.float32)
    else:
        actions_p0 = np.array(actions_p0_rows, dtype=np.float32)

    if actions_p1_rows is None or len(actions_p1_rows) == 0:
        actions_p1 = np.empty((0, 3), dtype=np.float32)
    else:
        actions_p1 = np.array(actions_p1_rows, dtype=np.float32)

    return StepRecord(
        turn=0,
        planets=planets,
        fleets=fleets,
        actions_p0=actions_p0,
        actions_p1=actions_p1,
        comet_planet_ids=np.empty(0, dtype=np.int32),
        is_terminal=False,
    )


# ---------------------------------------------------------------------------
# ActionContext
# ---------------------------------------------------------------------------

def test_action_context_empty():
    ctx = ActionContext(
        planet_ids=np.empty(0, dtype=np.int32),
        planet_positions=np.empty((0, 2), dtype=np.float32),
        my_planet_mask=np.empty(0, dtype=bool),
        n_planets=0,
    )
    assert ctx.n_planets == 0
    assert ctx.planet_ids.shape == (0,)
    assert ctx.planet_positions.shape == (0, 2)
    assert ctx.my_planet_mask.shape == (0,)


# ---------------------------------------------------------------------------
# ModelLabels
# ---------------------------------------------------------------------------

def test_model_labels_no_op():
    label = ModelLabels(action_type=0, source_idx=-1, target_idx=-1, amount_bin=-1, value_target=0.5)
    assert label.action_type == 0
    assert label.source_idx == -1
    assert label.target_idx == -1
    assert label.amount_bin == -1
    assert label.value_target == pytest.approx(0.5)


def test_model_labels_launch():
    label = ModelLabels(action_type=1, source_idx=2, target_idx=5, amount_bin=3, value_target=0.8)
    assert label.action_type == 1
    assert label.source_idx == 2
    assert label.target_idx == 5
    assert label.amount_bin == 3
    assert label.value_target == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# StateBuilder.input_dim
# ---------------------------------------------------------------------------

def test_state_builder_input_dim_default():
    sb = StateBuilder()
    assert sb.input_dim == 50 * 7 + 100 * 7  # 350 + 700 = 1050


def test_state_builder_input_dim_custom():
    sb = StateBuilder(max_planets=10, max_fleets=20)
    assert sb.input_dim == 10 * 7 + 20 * 7  # 70 + 140 = 210


# ---------------------------------------------------------------------------
# StateBuilder.from_step — standard case
# ---------------------------------------------------------------------------

def test_state_builder_from_step_standard():
    # Planet columns: [id, owner, x, y, radius, ships, production]
    # player 0 owns planet 10, player 1 owns planet 20
    planets = [
        [10, 0, 20.0, 30.0, 5.0, 100.0, 3.0],
        [20, 1, 60.0, 70.0, 5.0, 50.0,  2.0],
    ]
    # Fleet columns: [id, owner, x, y, angle, from_planet_id, ships]
    fleets = [
        [1, 0, 25.0, 35.0, math.pi / 4, 10, 30.0],
    ]
    step = _make_step(planets, fleets)
    sb = StateBuilder()
    result = sb.from_step(step, player=0)

    assert isinstance(result, ModelInput)
    assert result.array.shape == (1050,)
    assert result.array.dtype == np.float32
    assert result.context.n_planets == 2
    np.testing.assert_array_equal(result.context.planet_ids, np.array([10, 20], dtype=np.int32))
    # player 0 owns planet 10, so my_planet_mask[0] = True, [1] = False
    assert result.context.my_planet_mask[0] == True
    assert result.context.my_planet_mask[1] == False


def test_state_builder_from_step_planet_features():
    """Check specific feature values for a single planet owned by player 0."""
    planets = [
        [5, 0, 40.0, 80.0, 3.0, 100.0, 4.0],
    ]
    step = _make_step(planets, [])
    sb = StateBuilder()
    result = sb.from_step(step, player=0)

    arr = result.array
    # Planet 0 at offset 0
    assert arr[0] == pytest.approx(1.0)   # owner == player
    assert arr[1] == pytest.approx(0.0)   # opponent
    assert arr[2] == pytest.approx(0.0)   # neutral
    assert arr[3] == pytest.approx(40.0 / 100.0)
    assert arr[4] == pytest.approx(80.0 / 100.0)
    assert arr[5] == pytest.approx(min(100.0 / 200.0, 1.0))
    assert arr[6] == pytest.approx(4.0 / 5.0)


def test_state_builder_from_step_fleet_features():
    """Check specific feature values for a single fleet owned by player 1."""
    planets = [
        [1, 0, 10.0, 10.0, 2.0, 50.0, 1.0],
        [2, 1, 90.0, 90.0, 2.0, 50.0, 1.0],
    ]
    angle = math.pi / 3
    fleets = [
        [10, 1, 50.0, 50.0, angle, 2, 80.0],
    ]
    step = _make_step(planets, fleets)
    sb = StateBuilder()
    result = sb.from_step(step, player=0)

    arr = result.array
    fleet_base = sb.max_planets * 7  # 350
    # owner == 1, player == 0 → owner != player
    assert arr[fleet_base + 0] == pytest.approx(0.0)  # not mine
    assert arr[fleet_base + 1] == pytest.approx(1.0)  # opponent
    assert arr[fleet_base + 2] == pytest.approx(50.0 / 100.0)
    assert arr[fleet_base + 3] == pytest.approx(50.0 / 100.0)
    assert arr[fleet_base + 4] == pytest.approx(math.sin(angle))
    assert arr[fleet_base + 5] == pytest.approx(math.cos(angle))
    assert arr[fleet_base + 6] == pytest.approx(min(80.0 / 200.0, 1.0))


# ---------------------------------------------------------------------------
# StateBuilder.from_step — empty case
# ---------------------------------------------------------------------------

def test_state_builder_from_step_empty():
    step = _make_step([], [])
    sb = StateBuilder()
    result = sb.from_step(step, player=0)

    assert result.array.shape == (1050,)
    np.testing.assert_array_equal(result.array, np.zeros(1050, dtype=np.float32))
    assert result.context.n_planets == 0


# ---------------------------------------------------------------------------
# StateBuilder.__call__ matches from_step
# ---------------------------------------------------------------------------

def test_state_builder_call_matches_from_step():
    planets = [
        [1, 0, 10.0, 20.0, 3.0, 60.0, 2.0],
        [2, 1, 50.0, 60.0, 3.0, 40.0, 3.0],
    ]
    fleets = [
        [7, 0, 15.0, 25.0, 0.5, 1, 20.0],
    ]
    step = _make_step(planets, fleets)
    sb = StateBuilder()

    result_call = sb(step, player=0)
    result_from = sb.from_step(step, player=0)

    np.testing.assert_array_equal(result_call.array, result_from.array)
    np.testing.assert_array_equal(result_call.context.planet_ids, result_from.context.planet_ids)
    np.testing.assert_array_equal(result_call.context.my_planet_mask, result_from.context.my_planet_mask)
    assert result_call.context.n_planets == result_from.context.n_planets


# ---------------------------------------------------------------------------
# StateBuilder.from_obs matches from_step
# ---------------------------------------------------------------------------

def test_state_builder_from_obs_matches_from_step():
    planets_list = [
        [1, 0, 10.0, 20.0, 3.0, 60.0, 2.0],
        [2, 1, 50.0, 60.0, 3.0, 40.0, 3.0],
    ]
    fleets_list = [
        [7, 0, 15.0, 25.0, 0.5, 1, 20.0],
    ]
    obs = {"planets": planets_list, "fleets": fleets_list}
    step = _make_step(planets_list, fleets_list)
    sb = StateBuilder()

    result_obs = sb.from_obs(obs, player=0)
    result_step = sb.from_step(step, player=0)

    np.testing.assert_array_almost_equal(result_obs.array, result_step.array)
    np.testing.assert_array_equal(result_obs.context.planet_ids, result_step.context.planet_ids)


def test_state_builder_from_obs_empty():
    obs = {"planets": [], "fleets": []}
    sb = StateBuilder()
    result = sb.from_obs(obs, player=0)
    assert result.array.shape == (1050,)
    assert result.context.n_planets == 0


# ---------------------------------------------------------------------------
# ActionCodec.encode — empty actions
# ---------------------------------------------------------------------------

def test_action_codec_encode_empty():
    codec = ActionCodec()
    ctx = ActionContext(
        planet_ids=np.array([1, 2], dtype=np.int32),
        planet_positions=np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32),
        my_planet_mask=np.array([True, False]),
        n_planets=2,
    )
    planets = np.array([[1, 0, 10.0, 10.0, 3.0, 50.0, 2.0],
                        [2, 1, 90.0, 90.0, 3.0, 30.0, 3.0]], dtype=np.float32)
    raw_actions = np.empty((0, 3), dtype=np.float32)

    label = codec.encode(raw_actions, ctx, planets, value_target=0.6)
    assert label.action_type == ActionCodec.NO_OP
    assert label.source_idx == -1
    assert label.target_idx == -1
    assert label.amount_bin == -1
    assert label.value_target == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# ActionCodec.encode — valid LAUNCH
# ---------------------------------------------------------------------------

def test_action_codec_encode_launch():
    codec = ActionCodec()
    # planet 1 at (10,10) owned by player 0; planet 2 at (90,90) owned by player 1
    ctx = ActionContext(
        planet_ids=np.array([1, 2], dtype=np.int32),
        planet_positions=np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32),
        my_planet_mask=np.array([True, False]),
        n_planets=2,
    )
    planets = np.array([[1, 0, 10.0, 10.0, 3.0, 100.0, 2.0],
                        [2, 1, 90.0, 90.0, 3.0, 30.0,  3.0]], dtype=np.float32)
    # Action: from planet 1, angle pointing toward (90,90) from (10,10)
    angle = math.atan2(90.0 - 10.0, 90.0 - 10.0)  # pi/4
    raw_actions = np.array([[1, angle, 50.0]], dtype=np.float32)

    label = codec.encode(raw_actions, ctx, planets, value_target=0.7)
    assert label.action_type == ActionCodec.LAUNCH
    assert label.source_idx == 0       # planet_ids[0] == 1
    assert label.target_idx == 1       # only other planet is index 1
    assert 0 <= label.amount_bin < codec.n_amount_bins
    assert label.value_target == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# ActionCodec.encode — from_planet_id not in context
# ---------------------------------------------------------------------------

def test_action_codec_encode_unknown_planet():
    codec = ActionCodec()
    ctx = ActionContext(
        planet_ids=np.array([1, 2], dtype=np.int32),
        planet_positions=np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32),
        my_planet_mask=np.array([True, False]),
        n_planets=2,
    )
    planets = np.array([[1, 0, 10.0, 10.0, 3.0, 100.0, 2.0],
                        [2, 1, 90.0, 90.0, 3.0, 30.0,  3.0]], dtype=np.float32)
    raw_actions = np.array([[99, 0.5, 40.0]], dtype=np.float32)  # planet 99 doesn't exist

    label = codec.encode(raw_actions, ctx, planets, value_target=0.5)
    assert label.action_type == ActionCodec.NO_OP
    assert label.source_idx == -1


# ---------------------------------------------------------------------------
# ActionCodec.decode — real implementation (Cycle 2)
# ---------------------------------------------------------------------------

def test_action_codec_decode_empty_context_returns_no_op():
    """decode returns [] when there are no planets."""
    codec = ActionCodec()
    ctx = ActionContext(
        planet_ids=np.empty(0, dtype=np.int32),
        planet_positions=np.empty((0, 2), dtype=np.float32),
        my_planet_mask=np.empty(0, dtype=bool),
        n_planets=0,
    )
    planets = np.empty((0, 7), dtype=np.float32)
    result = codec.decode(None, ctx, planets)
    assert result == []


def test_action_codec_decode_noop_when_action_type_is_zero():
    """decode returns [] when action_type_logits strongly favors NO_OP (index 0)."""
    codec = ActionCodec()
    ctx = ActionContext(
        planet_ids=np.array([10, 20], dtype=np.int32),
        planet_positions=np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32),
        my_planet_mask=np.array([True, False]),
        n_planets=2,
    )
    planets = np.array(
        [[10, 0, 10.0, 10.0, 3.0, 100.0, 2.0],
         [20, 1, 90.0, 90.0, 3.0,  30.0, 3.0]],
        dtype=np.float32,
    )
    output = PolicyOutput(
        action_type_logits=np.array([100.0, -100.0]),
        source_logits=np.array([1.0, 0.0]),
        target_logits=np.array([0.0, 1.0]),
        amount_logits=np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        value=np.array([0.0]),
    )
    result = codec.decode(output, ctx, planets)
    assert result == []


def test_action_codec_decode_launch_returns_action():
    """decode returns a valid LAUNCH action when action_type favors LAUNCH."""
    codec = ActionCodec()
    ctx = ActionContext(
        planet_ids=np.array([10, 20], dtype=np.int32),
        planet_positions=np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32),
        my_planet_mask=np.array([True, False]),
        n_planets=2,
    )
    planets = np.array(
        [[10, 0, 10.0, 10.0, 3.0, 100.0, 2.0],
         [20, 1, 90.0, 90.0, 3.0,  30.0, 3.0]],
        dtype=np.float32,
    )
    output = PolicyOutput(
        action_type_logits=np.array([-100.0, 100.0]),
        source_logits=np.array([100.0, -100.0]),
        target_logits=np.array([-100.0, 100.0]),
        amount_logits=np.array([0.0, 0.0, 100.0, 0.0, 0.0]),  # bin 2 → 0.5
        value=np.array([0.0]),
    )
    result = codec.decode(output, ctx, planets)
    assert len(result) == 1
    assert result[0][0] == 10                           # source planet_id
    assert result[0][2] == pytest.approx(0.5 * 100.0)  # 50 ships
    assert isinstance(result[0][1], float)              # angle is a float


def test_action_codec_decode_source_mask_excludes_non_owned():
    """decode picks the owned planet even when a non-owned planet has higher raw logit."""
    codec = ActionCodec()
    ctx = ActionContext(
        planet_ids=np.array([10, 20, 30], dtype=np.int32),
        planet_positions=np.array(
            [[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]], dtype=np.float32
        ),
        my_planet_mask=np.array([False, True, False]),
        n_planets=3,
    )
    planets = np.array(
        [[10, 1, 10.0, 10.0, 3.0,  80.0, 2.0],
         [20, 0, 50.0, 50.0, 3.0, 120.0, 2.0],
         [30, 1, 90.0, 90.0, 3.0,  60.0, 2.0]],
        dtype=np.float32,
    )
    output = PolicyOutput(
        action_type_logits=np.array([-100.0, 100.0]),
        source_logits=np.array([100.0, 50.0, -100.0]),  # raw argmax is index 0 (not owned)
        target_logits=np.array([-100.0, -100.0, 100.0]),
        amount_logits=np.array([0.0, 0.0, 100.0, 0.0, 0.0]),
        value=np.array([0.0]),
    )
    result = codec.decode(output, ctx, planets)
    assert len(result) == 1
    assert result[0][0] == 20  # planet_ids[1] — the owned planet was selected


def test_action_codec_decode_no_ships_returns_empty():
    """decode returns [] when the source planet has zero ships."""
    codec = ActionCodec()
    ctx = ActionContext(
        planet_ids=np.array([10, 20], dtype=np.int32),
        planet_positions=np.array([[10.0, 10.0], [90.0, 90.0]], dtype=np.float32),
        my_planet_mask=np.array([True, False]),
        n_planets=2,
    )
    planets = np.array(
        [[10, 0, 10.0, 10.0, 3.0, 0.0, 2.0],   # source planet has 0 ships
         [20, 1, 90.0, 90.0, 3.0, 30.0, 3.0]],
        dtype=np.float32,
    )
    output = PolicyOutput(
        action_type_logits=np.array([-100.0, 100.0]),
        source_logits=np.array([100.0, -100.0]),
        target_logits=np.array([-100.0, 100.0]),
        amount_logits=np.array([100.0, 0.0, 0.0, 0.0, 0.0]),  # bin 0 → 0.1
        value=np.array([0.0]),
    )
    result = codec.decode(output, ctx, planets)
    assert result == []


# ---------------------------------------------------------------------------
# StateBuilder.from_step_structured
# ---------------------------------------------------------------------------

def test_state_builder_structured_shapes():
    planets = [
        [10, 0, 20.0, 30.0, 5.0, 100.0, 3.0],
        [20, 1, 60.0, 70.0, 5.0, 50.0, 2.0],
    ]
    fleets = [
        [1, 0, 25.0, 35.0, math.pi / 4, 10, 30.0],
    ]
    step = _make_step(planets, fleets)
    sb = StateBuilder(max_planets=50, max_fleets=100)
    result = sb.from_step_structured(step, player=0)

    assert result["planet_features"].shape == (50, 7)
    assert result["fleet_features"].shape == (700,)
    assert result["planet_mask"].shape == (50,)
    assert result["planet_features"].dtype == np.float32
    assert result["fleet_features"].dtype == np.float32
    assert result["planet_mask"].dtype == bool


def test_state_builder_structured_planet_mask_n_real():
    """planet_mask has exactly n_planets True values."""
    planets = [
        [10, 0, 20.0, 30.0, 5.0, 100.0, 3.0],
        [20, 1, 60.0, 70.0, 5.0, 50.0, 2.0],
        [30, -1, 50.0, 50.0, 5.0, 80.0, 1.0],
    ]
    step = _make_step(planets, [])
    sb = StateBuilder(max_planets=50)
    result = sb.from_step_structured(step, player=0)

    assert result["planet_mask"].sum() == 3
    assert np.all(result["planet_mask"][:3])
    assert not np.any(result["planet_mask"][3:])


def test_state_builder_structured_planet_features_values():
    """Planet feature values match the same normalisation as from_step."""
    planets = [
        [5, 0, 40.0, 80.0, 3.0, 100.0, 4.0],
    ]
    step = _make_step(planets, [])
    sb = StateBuilder()
    structured = sb.from_step_structured(step, player=0)

    pf = structured["planet_features"]
    assert pf[0, 0] == pytest.approx(1.0)           # owner_self
    assert pf[0, 1] == pytest.approx(0.0)           # owner_enemy
    assert pf[0, 2] == pytest.approx(0.0)           # owner_neutral
    assert pf[0, 3] == pytest.approx(40.0 / 100.0) # x
    assert pf[0, 4] == pytest.approx(80.0 / 100.0) # y
    assert pf[0, 5] == pytest.approx(min(100.0 / 200.0, 1.0))  # ships
    assert pf[0, 6] == pytest.approx(4.0 / 5.0)    # production


def test_state_builder_structured_padding_is_zero():
    """Padding slots in planet_features must be zero-filled."""
    planets = [[10, 0, 20.0, 30.0, 5.0, 50.0, 2.0]]
    step = _make_step(planets, [])
    sb = StateBuilder(max_planets=10)
    result = sb.from_step_structured(step, player=0)

    # Slots 1..9 should all be zero
    np.testing.assert_array_equal(
        result["planet_features"][1:], np.zeros((9, 7), dtype=np.float32)
    )


def test_state_builder_structured_context_matches_from_step():
    """context returned by from_step_structured equals from_step context."""
    planets = [
        [10, 0, 20.0, 30.0, 5.0, 100.0, 3.0],
        [20, 1, 60.0, 70.0, 5.0, 50.0, 2.0],
    ]
    step = _make_step(planets, [])
    sb = StateBuilder()

    flat = sb.from_step(step, player=0)
    structured = sb.from_step_structured(step, player=0)

    np.testing.assert_array_equal(
        structured["context"].planet_ids, flat.context.planet_ids
    )
    np.testing.assert_array_equal(
        structured["context"].my_planet_mask, flat.context.my_planet_mask
    )
    assert structured["context"].n_planets == flat.context.n_planets


def test_state_builder_structured_empty():
    step = _make_step([], [])
    sb = StateBuilder()
    result = sb.from_step_structured(step, player=0)

    assert result["planet_features"].shape == (50, 7)
    assert result["fleet_features"].shape == (700,)
    np.testing.assert_array_equal(result["planet_mask"], np.zeros(50, dtype=bool))


def test_state_builder_from_obs_structured_matches_from_step_structured():
    planets_list = [[1, 0, 10.0, 20.0, 3.0, 60.0, 2.0], [2, 1, 50.0, 60.0, 3.0, 40.0, 3.0]]
    fleets_list = [[7, 0, 15.0, 25.0, 0.5, 1, 20.0]]
    obs = {"planets": planets_list, "fleets": fleets_list}
    step = _make_step(planets_list, fleets_list)
    sb = StateBuilder()

    r_obs = sb.from_obs_structured(obs, player=0)
    r_step = sb.from_step_structured(step, player=0)

    np.testing.assert_array_almost_equal(r_obs["planet_features"], r_step["planet_features"])
    np.testing.assert_array_almost_equal(r_obs["fleet_features"], r_step["fleet_features"])
    np.testing.assert_array_equal(r_obs["planet_mask"], r_step["planet_mask"])
