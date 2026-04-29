import pytest
from tournament.elo import update_elo


def test_elo_equal_ratings_zero_sum():
    ratings = {"A": 1500, "B": 1500}
    new = update_elo(ratings, "A", "B")
    assert new["A"] + new["B"] == pytest.approx(3000.0)


def test_elo_equal_ratings_winner_gains_16():
    ratings = {"A": 1500, "B": 1500}
    new = update_elo(ratings, "A", "B", k=32)
    assert new["A"] - 1500 == pytest.approx(16.0)
    assert new["B"] - 1500 == pytest.approx(-16.0)


def test_elo_higher_rated_wins_smaller_gain():
    ratings = {"A": 1700, "B": 1300}
    new = update_elo(ratings, "A", "B", k=32)
    assert new["A"] - 1700 < 16.0


def test_elo_lower_rated_upsets_bigger_gain():
    ratings = {"A": 1300, "B": 1700}
    new = update_elo(ratings, "A", "B", k=32)
    assert new["A"] - 1300 > 16.0


def test_elo_returns_new_dict_does_not_mutate():
    original = {"A": 1500, "B": 1500}
    old_a = original["A"]
    update_elo(original, "A", "B")
    assert original["A"] == old_a


def test_elo_unknown_player_raises_keyerror():
    ratings = {"A": 1500}
    with pytest.raises(KeyError):
        update_elo(ratings, "A", "Z")
