"""CLI entry point for PPO-based reinforcement learning training.

Usage:
    python scripts/train_rl.py --config training/rl_config.json
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training.utils.rl_config import RLConfig
from bots.neural.planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec
from training.trainers.rl_trainer import RLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Orbit Wars bot with PPO")
    parser.add_argument("--config", required=True, help="Path to RLConfig JSON")
    args = parser.parse_args()

    config = RLConfig.from_json(Path(args.config))

    model_config_dict = config.model_config
    planet_cfg = PlanetPolicyConfig(
        Dp=model_config_dict.get("Dp", 10),
        Df=model_config_dict.get("Df", 8),
        Dg=model_config_dict.get("Dg", 4),
        E=model_config_dict.get("E", 64),
        F=model_config_dict.get("F", 32),
        G=model_config_dict.get("G", 128),
        max_planets=model_config_dict.get("max_planets", 50),
        max_fleets=model_config_dict.get("max_fleets", 200),
        n_amount_bins=model_config_dict.get("n_amount_bins", 5),
        dropout=model_config_dict.get("dropout", 0.1),
        n_attn_heads=model_config_dict.get("n_attn_heads", 2),
    )
    model = PlanetPolicyModel(planet_cfg)

    if config.frozen_checkpoint:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_path = (
            config.frozen_checkpoint
            if os.path.isabs(config.frozen_checkpoint)
            else os.path.join(root, config.frozen_checkpoint)
        )
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded weights from: {ckpt_path}")

    state_builder = StateBuilder(
        max_planets=planet_cfg.max_planets,
        max_fleets=planet_cfg.max_fleets,
    )
    codec = ActionCodec(n_amount_bins=planet_cfg.n_amount_bins)

    RLTrainer(config, model, state_builder, codec).train()


if __name__ == "__main__":
    main()
