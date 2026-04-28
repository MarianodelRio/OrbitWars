"""RLConfig: configuration dataclass for PPO-based RL training."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from training.utils.device import resolve_device


@dataclass
class RLConfig:
    # Rollout
    n_rollout_steps: int = 2048
    n_envs: int = 1
    steps_per_episode: int = 500

    # PPO
    ppo_epochs: int = 4
    ppo_batch_size: int = 256
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    normalize_advantages: bool = True

    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Reward
    w_planets: float = 1.0
    w_production: float = 0.5
    w_ships: float = 0.1
    reward_lambda: float = 0.05
    reward_clip_abs: float = 0.2

    # Opponent pool
    max_snapshots: int = 5
    snapshot_every: int = 50
    heuristic_opponents: list = field(
        default_factory=lambda: [
            "bots.heuristic.baseline:agent_fn",
        ]
    )
    frozen_checkpoint: str | None = None
    self_play_prob: float = 0.3

    # Evaluation
    eval_every: int = 100
    n_eval_matches: int = 10
    eval_opponents: list = field(default_factory=lambda: ["heuristic.baseline"])

    # Checkpointing
    save_every: int = 100

    # LR schedule
    lr_schedule: str = "cosine"

    # Run metadata
    run_name: str = "rl_run"
    run_id: str = ""
    device: str = "cpu"
    seed: int = 42
    total_iterations: int = 1000

    # Model config (optional dict for building PlanetPolicyModel)
    model_config: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "RLConfig":
        with open(path, "r") as f:
            data = json.load(f)
        defaults = {f.name: f.default if f.default is not dataclasses.MISSING else f.default_factory() for f in dataclasses.fields(cls)}
        for f in dataclasses.fields(cls):
            if f.name not in data:
                if f.default is not dataclasses.MISSING:
                    data[f.name] = f.default
                elif f.default_factory is not dataclasses.MISSING:
                    data[f.name] = f.default_factory()
        # Only pass known fields
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        if "device" in filtered:
            filtered["device"] = resolve_device(filtered["device"])
        return cls(**filtered)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / "rl_config.json"
        with open(out_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @property
    def run_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent / "runs" / self.run_name / self.run_id
