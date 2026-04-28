"""CLI entry point for imitation learning training.

Usage:
    python scripts/train_il.py --config training/il_config.json
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training.utils.run_config import RunConfig
from training.utils.device import resolve_device
from bots.neural.planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel
from bots.neural.state_builder import StateBuilder
from bots.neural.action_codec import ActionCodec
from training.trainers.il_trainer import ILTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to RunConfig JSON")
    parser.add_argument("--device", default=None, help="Override device: cpu, cuda, or auto")
    args = parser.parse_args()

    config = RunConfig.from_json(Path(args.config))
    if args.device is not None:
        config.device = resolve_device(args.device)

    model_type = config.model_config.get("model_type", "planet_policy")

    if model_type == "planet_policy":
        model_config_dict = config.model_config
        planet_cfg = PlanetPolicyConfig(
            Dp=model_config_dict.get("Dp", 24),
            Df=model_config_dict.get("Df", 16),
            Dg=model_config_dict.get("Dg", 16),
            E=model_config_dict.get("E", 192),
            F=model_config_dict.get("F", 128),
            G=model_config_dict.get("G", 384),
            max_planets=model_config_dict.get("max_planets", 50),
            max_fleets=model_config_dict.get("max_fleets", 200),
            n_amount_bins=model_config_dict.get("n_amount_bins", 8),
            dropout=model_config_dict.get("dropout", 0.1),
            n_heads=model_config_dict.get("n_heads", 8),
            lstm_bypass=model_config_dict.get("lstm_bypass", False),
        )
        model = PlanetPolicyModel(planet_cfg)
        state_builder = StateBuilder(max_planets=planet_cfg.max_planets, max_fleets=planet_cfg.max_fleets)
        angular_diff_threshold = getattr(config, "angular_diff_threshold", 0.7853981633974483)
        codec = ActionCodec(n_amount_bins=planet_cfg.n_amount_bins, angular_diff_threshold=angular_diff_threshold)
    else:
        print(f"Error: unsupported model_type {model_type!r}. Only 'planet_policy' is supported.")
        sys.exit(1)

    if config.resume_from:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_path = config.resume_from if os.path.isabs(config.resume_from) else os.path.join(_root, config.resume_from)
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed from: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    ILTrainer(config, model, state_builder, codec).train()


if __name__ == "__main__":
    main()
