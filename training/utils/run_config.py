"""RunConfig dataclass for training run configuration."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.utils.device import resolve_device


@dataclass
class RunConfig:
    run_name: str
    run_id: str
    model_config: dict
    lr: float
    batch_size: int
    epochs: int
    val_split: float
    eval_every: int
    eval_opponents: list
    n_eval_matches: int
    data_pipeline: dict
    device: str
    seed: int
    weight_decay: float = 1e-4
    action_type_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    use_class_weights: bool = True
    resume_from: str | None = None
    angular_diff_threshold: float = 0.7853981633974483  # π/4
    lr_schedule: str = "constant"
    early_stopping_patience: int = 0

    @classmethod
    def from_json(cls, path: Path) -> "RunConfig":
        with open(path, "r") as f:
            data = json.load(f)

        if "run_id" not in data or not data.get("run_id"):
            run_name = data.get("run_name", "run")
            run_dir_parent = (
                Path(__file__).resolve().parent.parent.parent
                / "runs"
                / run_name
            )
            data["run_id"] = cls._next_run_id(run_dir_parent)

        data["device"] = resolve_device(data["device"])
        return cls(**data)

    @property
    def run_dir(self) -> Path:
        return (
            Path(__file__).resolve().parent.parent.parent
            / "runs"
            / self.run_name
            / self.run_id
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @staticmethod
    def _next_run_id(run_dir_parent: Path) -> str:
        if not run_dir_parent.exists():
            return "run_001"

        existing = [
            d.name for d in run_dir_parent.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

        max_num = 0
        for name in existing:
            suffix = name[4:]
            if suffix.isdigit():
                num = int(suffix)
                if num > max_num:
                    max_num = num

        return f"run_{max_num + 1:03d}"
