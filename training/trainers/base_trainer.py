"""BaseTrainer ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from bots.neural.planet_policy_model import PlanetPolicyModel
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec
from training.utils.run_config import RunConfig
from training.utils.checkpointing import CheckpointManager
from training.utils.metrics import MetricsLogger


class BaseTrainer(ABC):
    def __init__(
        self,
        config: RunConfig,
        model: PlanetPolicyModel,
        state_builder: StateBuilder,
        codec: ActionCodec,
    ) -> None:
        self.config = config
        self.model = model
        self.state_builder = state_builder
        self.codec = codec
        self._ckpt_manager: CheckpointManager | None = None
        self._train_logger: MetricsLogger | None = None
        self._val_logger: MetricsLogger | None = None

    @abstractmethod
    def train(self) -> None:
        ...

    def _setup_run_dir(self) -> None:
        run_dir = self.config.run_dir
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)

        self.config.save(run_dir)

        self._ckpt_manager = CheckpointManager(run_dir)

        fields = ["epoch", "loss"]
        self._train_logger = MetricsLogger(run_dir / "metrics" / "train.csv", fields)
        self._val_logger = MetricsLogger(run_dir / "metrics" / "val.csv", fields)

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        self._ckpt_manager.save(
            self.model,
            self.state_builder,
            self.codec,
            epoch,
            metrics,
            is_best,
        )

    def _log_train(self, row: dict) -> None:
        self._train_logger.log(row)

    def _log_val(self, row: dict) -> None:
        self._val_logger.log(row)
