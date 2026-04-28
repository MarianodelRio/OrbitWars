import pytest
from game.env.runner import run_match
from bots.heuristic.sniper import agent_fn as sniper
from bots.heuristic.baseline import agent_fn as baseline
from bots.heuristic.sniper import agent_fn as baseline_bot

def test_sniper_vs_baseline_runs():
    result = run_match(sniper, baseline, steps=200)
    assert "winner" in result
    assert len(result["rewards"]) == 2

def test_sniper_beats_baseline_majority():
    wins = 0
    n = 10
    for _ in range(n):
        result = run_match(sniper, baseline, steps=200)
        if result["winner"] == 0:
            wins += 1
    assert wins >= 6, f"Sniper only won {wins}/{n} — expected >=6"

def test_baseline_bot_runs():
    result = run_match(baseline_bot, baseline, steps=100)
    assert "winner" in result
