"""Elo rating computation."""


def update_elo(ratings: dict, winner: str, loser: str, k: int = 32) -> dict:
    """Update Elo ratings after a match. Returns updated ratings dict."""
    ra, rb = ratings[winner], ratings[loser]
    expected_a = 1 / (1 + 10 ** ((rb - ra) / 400))
    ratings = dict(ratings)
    ratings[winner] = ra + k * (1 - expected_a)
    ratings[loser] = rb + k * (0 - (1 - expected_a))
    return ratings
