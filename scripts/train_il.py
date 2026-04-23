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
from bots.neural.model import PolicyValueModel, PolicyValueConfig
from bots.neural.pointer_model import PointerNetworkModel, PointerNetworkConfig
from bots.neural.planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel
from bots.neural.state_builder import StateBuilder
from bots.neural.state_builder_v2 import StateBuilderV2
from bots.neural.action_codec import ActionCodec
from bots.neural.action_codec_v2 import ActionCodecV2
from training.trainers.il_trainer import ILTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to RunConfig JSON")
    args = parser.parse_args()

    config = RunConfig.from_json(Path(args.config))

    model_type = config.model_config.get("model_type", "flat")

    if model_type == "planet_policy":
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
        state_builder = StateBuilderV2(max_planets=planet_cfg.max_planets, max_fleets=planet_cfg.max_fleets)
        angular_diff_threshold = getattr(config, "angular_diff_threshold", 0.7853981633974483)
        codec = ActionCodecV2(n_amount_bins=planet_cfg.n_amount_bins, angular_diff_threshold=angular_diff_threshold)
    elif model_type == "pointer":
        # Remove model_type before passing to dataclass
        cfg_dict = {k: v for k, v in config.model_config.items() if k != "model_type"}
        model_cfg = PointerNetworkConfig(**cfg_dict)
        model = PointerNetworkModel(model_cfg)
        max_planets = model_cfg.max_planets
        max_fleets = model_cfg.max_fleets
        n_amount_bins = model_cfg.n_amount_bins
        state_builder = StateBuilder(max_planets=max_planets, max_fleets=max_fleets)
        codec = ActionCodec(n_amount_bins=n_amount_bins)
    else:
        cfg_dict = {k: v for k, v in config.model_config.items() if k != "model_type"}
        model_cfg = PolicyValueConfig(**cfg_dict)
        model = PolicyValueModel(model_cfg)
        max_planets = model_cfg.max_planets
        max_fleets = getattr(model_cfg, "max_fleets", 100)
        n_amount_bins = model_cfg.n_amount_bins
        state_builder = StateBuilder(max_planets=max_planets, max_fleets=max_fleets)
        codec = ActionCodec(n_amount_bins=n_amount_bins)

    if config.resume_from:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ckpt_path = config.resume_from if os.path.isabs(config.resume_from) else os.path.join(_root, config.resume_from)
        ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed from: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

    ILTrainer(config, model, state_builder, codec).train()


if __name__ == "__main__":
    main()
