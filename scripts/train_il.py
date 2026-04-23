"""CLI entry point for imitation learning training.

Usage:
    python scripts/train_il.py --config training/il_config.json
"""

import argparse
from pathlib import Path

from training.utils.run_config import RunConfig
from bots.neural.model import PolicyValueModel, PolicyValueConfig
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec
from training.trainers.il_trainer import ILTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to RunConfig JSON")
    args = parser.parse_args()

    config = RunConfig.from_json(Path(args.config))

    model_cfg = PolicyValueConfig(**config.model_config)
    model = PolicyValueModel(model_cfg)

    state_builder = StateBuilder(
        max_planets=model_cfg.max_planets,
        max_fleets=model_cfg.max_fleets if hasattr(model_cfg, "max_fleets") else 100,
    )
    codec = ActionCodec(n_amount_bins=model_cfg.n_amount_bins)

    ILTrainer(config, model, state_builder, codec).train()


if __name__ == "__main__":
    main()
