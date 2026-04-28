"""Unified CLI entry point for Orbit Wars training and evaluation.

Usage examples:
    # RL training
    python train.py --config training/rl_config.json

    # IL training
    python train.py --config training/il_config.json

    # Dry run (print mode/config, do not train)
    python train.py --config training/rl_config.json --dry-run

    # Evaluation
    python train.py eval --checkpoint runs/.../checkpoints/rl_last.pt

    # Evaluation with specific opponents and match count
    python train.py eval --checkpoint path/to/ckpt.pt --opponents heuristic.baseline scoring.bot --n-matches 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _detect_mode(data: dict, explicit_mode: str | None) -> str:
    if explicit_mode:
        return explicit_mode
    if "total_iterations" in data and "n_rollout_steps" in data:
        return "rl"
    if "epochs" in data:
        return "il"
    raise ValueError(
        "Cannot detect training mode from config. "
        "Add a 'mode' key ('rl' or 'il'), or include 'total_iterations'+'n_rollout_steps' for RL "
        "or 'epochs' for IL."
    )


def _run_training(args) -> None:
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        data = json.load(f)

    mode = _detect_mode(data, data.get("mode"))

    if mode == "rl":
        from training.utils.rl_config import RLConfig
        from training.utils.device import resolve_device
        from bots.neural.planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel
        from bots.neural.state_builder import StateBuilder
        from bots.neural.action_codec import ActionCodec
        from training.trainers.rl_trainer import RLTrainer

        config = RLConfig.from_json(config_path)
        if args.device is not None:
            config.device = resolve_device(args.device)

        if args.dry_run:
            print(f"mode     : rl")
            print(f"run_name : {config.run_name}")
            print(f"device   : {config.device}")
            print(f"run_dir  : {config.run_dir}")
            print(f"opponents: {config.heuristic_opponents}")
            sys.exit(0)

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
        )
        model = PlanetPolicyModel(planet_cfg)

        import os
        import torch
        if config.frozen_checkpoint:
            root = str(_REPO_ROOT)
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

    elif mode == "il":
        from training.utils.run_config import RunConfig
        from training.utils.device import resolve_device
        from bots.neural.planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel
        from bots.neural.state_builder import StateBuilder
        from bots.neural.action_codec import ActionCodec
        from training.trainers.il_trainer import ILTrainer

        import os
        import torch

        config = RunConfig.from_json(config_path)
        if args.device is not None:
            config.device = resolve_device(args.device)

        if args.dry_run:
            print(f"mode     : il")
            print(f"run_name : {config.run_name}")
            print(f"device   : {config.device}")
            print(f"run_dir  : {config.run_dir}")
            print(f"opponents: {config.eval_opponents}")
            sys.exit(0)

        model_type = config.model_config.get("model_type", "planet_policy")
        if model_type != "planet_policy":
            print(f"Error: unsupported model_type {model_type!r}. Only 'planet_policy' is supported.")
            sys.exit(1)

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
        )
        model = PlanetPolicyModel(planet_cfg)
        state_builder = StateBuilder(max_planets=planet_cfg.max_planets, max_fleets=planet_cfg.max_fleets)
        angular_diff_threshold = getattr(config, "angular_diff_threshold", 0.7853981633974483)
        codec = ActionCodec(n_amount_bins=planet_cfg.n_amount_bins, angular_diff_threshold=angular_diff_threshold)

        if config.resume_from:
            root = str(_REPO_ROOT)
            ckpt_path = config.resume_from if os.path.isabs(config.resume_from) else os.path.join(root, config.resume_from)
            ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            print(f"Resumed from: {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")

        ILTrainer(config, model, state_builder, codec).train()

    else:
        print(f"Error: unknown mode {mode!r}. Expected 'rl' or 'il'.")
        sys.exit(1)


def _run_eval(args) -> None:
    from bots.neural.bot import NeuralBot
    from bots.registry import list_bots, resolve
    from training.evaluation.evaluator import Evaluator

    checkpoint_path = Path(args.checkpoint)
    bot = NeuralBot.load(str(checkpoint_path))

    if args.opponents:
        opponents = args.opponents
    else:
        opponents = list_bots()

    evaluator = Evaluator(
        bot=bot,
        opponents=opponents,
        n_matches=args.n_matches,
        run_dir=None,
    )
    results = evaluator.run(epoch=None)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orbit Wars unified training and evaluation CLI"
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint against opponents")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    eval_parser.add_argument(
        "--opponents",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Space-separated opponent names from registry (default: all registered bots)",
    )
    eval_parser.add_argument("--n-matches", type=int, default=20, help="Number of matches per opponent")

    # default training mode args (when no subcommand)
    parser.add_argument("--config", default=None, help="Path to config JSON (RL or IL)")
    parser.add_argument("--device", default=None, help="Override device: cpu, cuda, or auto")
    parser.add_argument("--dry-run", action="store_true", help="Print config info and exit without training")

    args = parser.parse_args()

    if args.subcommand == "eval":
        _run_eval(args)
    else:
        if args.config is None:
            parser.print_help()
            sys.exit(1)
        _run_training(args)


if __name__ == "__main__":
    main()
