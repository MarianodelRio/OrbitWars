"""RLTrainer placeholder."""

from __future__ import annotations

from training.trainers.base_trainer import BaseTrainer


class RLTrainer(BaseTrainer):
    """Placeholder — not implemented.

    When implemented, will use:
    - OrbitWarsEnv (training/envs/gym_wrapper.py) — step() pending
    - shaped_reward (training/rewards/shaped.py)
    - CheckpointManager (same as ILTrainer)
    - Evaluator (same as ILTrainer)
    - MetricsLogger (same as ILTrainer)
    """

    def train(self) -> None:
        raise NotImplementedError("RLTrainer not implemented yet.")
