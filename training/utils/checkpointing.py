"""CheckpointManager: saves and loads training checkpoints."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch

import dataclasses

from bots.neural.bot import NeuralBot
from bots.neural.pointer_model import PointerNetworkModel
from bots.neural.planet_policy_model import PlanetPolicyModel


class CheckpointManager:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._ckpt_dir = run_dir / "checkpoints"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def save(
        self,
        model,
        state_builder,
        codec,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
    ) -> Path:
        if isinstance(model, PlanetPolicyModel):
            model_type = "planet_policy"
        elif isinstance(model, PointerNetworkModel):
            model_type = "pointer"
        else:
            model_type = "flat"

        if isinstance(model, PlanetPolicyModel):
            config_to_save = dataclasses.asdict(model.config)
        else:
            config_to_save = model.config

        checkpoint = {
            "model_type": model_type,
            "config": config_to_save,
            "config_dict": model.config.__dict__ if not isinstance(model, PlanetPolicyModel) else config_to_save,
            "state_dict": model.state_dict(),
            "max_planets": state_builder.max_planets,
            "max_fleets": state_builder.max_fleets,
            "n_amount_bins": codec.n_amount_bins,
            "epoch": epoch,
            "train_loss": metrics.get("train_loss"),
            "val_loss": metrics.get("val_loss"),
            "timestamp": datetime.utcnow().isoformat(),
        }

        epoch_path = self._ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)

        last_path = self._ckpt_dir / "last.pt"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self._ckpt_dir / "best.pt"
            torch.save(checkpoint, best_path)

        return epoch_path

    def load_bot(self, tag: str = "best", device: str = "cpu") -> NeuralBot:
        path = self._ckpt_dir / f"{tag}.pt"
        return NeuralBot.load(str(path), device=device)

    def list_checkpoints(self) -> list[Path]:
        excluded = {"best.pt", "last.pt"}
        paths = [
            p for p in self._ckpt_dir.glob("*.pt")
            if p.name not in excluded
        ]
        return sorted(paths)
