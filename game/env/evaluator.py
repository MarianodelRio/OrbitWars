import importlib
import os
from game.env.runner import run_match

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_agent(path_str):
    """Load agent from 'module.path:attr' or 'module.path:attr?checkpoint=<path>' string.

    When ?checkpoint=<path> is provided the agent is a lazy-loading closure that
    calls NeuralBot.load() on the first invocation.  Relative checkpoint paths
    are resolved from the repository root.
    """
    checkpoint = None
    if "?checkpoint=" in path_str:
        path_str, qs = path_str.split("?checkpoint=", 1)
        checkpoint = qs.strip()

    module_path, attr = path_str.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    fn = getattr(mod, attr)

    if checkpoint is None:
        return fn

    ckpt_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(_REPO_ROOT, checkpoint)
    _bot = [None]

    def _agent(obs, config=None):
        if _bot[0] is None:
            from bots.neural.bot import NeuralBot
            print(f"[load_agent] Loading checkpoint: {ckpt_path}")
            _bot[0] = NeuralBot.load(ckpt_path)
        return _bot[0].act(obs, config)

    _agent.__name__ = f"neural({os.path.basename(ckpt_path)})"
    return _agent

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
