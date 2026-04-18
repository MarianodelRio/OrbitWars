import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.env.runner import run_match
from bots.sniper_bot import agent_fn as sniper
from bots.random_bot import agent_fn as random_agent

result = run_match(sniper, random_agent, steps=200, render=True, output_file="game.html")
print(f"Winner: {result['winner']}")
print(f"Rewards: {result['rewards']}")
print(f"Steps: {result['steps']}")
print(f"HTML saved to: game.html")
