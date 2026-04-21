import importlib
import os
from game.env.runner import run_match

def load_agent(path_str):
    """Load agent from 'module.path:attr' string."""
    module_path, attr = path_str.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)

def evaluate(bot1, bot2, n_matches=10, steps=500, save_data=False, data_dir=None):
    wins = [0, 0]
    draws = 0
    total_ships = [0.0, 0.0]
    total_steps = 0
    data_paths = []
    for match_idx in range(n_matches):
        if save_data and data_dir is not None:
            match_data_path = os.path.join(data_dir, f"match_{match_idx:04d}.h5")
        else:
            match_data_path = None
        result = run_match(bot1, bot2, steps=steps, render=False, save_data=save_data, data_path=match_data_path)
        data_paths.append(match_data_path)
        if result["winner"] is None:
            draws += 1
        else:
            wins[result["winner"]] += 1
        total_ships[0] += result["rewards"][0]
        total_ships[1] += result["rewards"][1]
        total_steps += result["steps"]
    return {
        "win_rate": [wins[i] / n_matches for i in range(2)],
        "avg_ships": [total_ships[i] / n_matches for i in range(2)],
        "avg_game_length": total_steps / n_matches,
        "draws": draws,
        "wins": wins,
        "data_paths": data_paths,
    }
