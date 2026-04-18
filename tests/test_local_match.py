import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.env.runner import run_match
from bots.baseline_bot import agent_fn as baseline


def test_match_completes():
    result = run_match(baseline, baseline, steps=10, render=False)
    assert "winner" in result
    assert "rewards" in result
    assert "steps" in result
    assert result["steps"] > 0
