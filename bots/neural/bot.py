"""NeuralBot — inference wrapper around PolicyValueModel or PointerNetworkModel."""

from __future__ import annotations

import numpy as np
import torch

from bots.interface import Bot
from .action_codec import ActionCodec
from .model import PolicyOutput, PolicyValueConfig, PolicyValueModel
from .pointer_model import PointerNetworkConfig, PointerNetworkModel, PointerPolicyOutput
from .planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel
from .state_builder import StateBuilder
from .state_builder_v2 import StateBuilderV2
from .action_codec_v2 import ActionCodecV2


class NeuralBot(Bot):
    def __init__(
        self,
        model: PolicyValueModel,
        state_builder: StateBuilder,
        codec: ActionCodec,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.state_builder = state_builder
        self.codec = codec
        self.device = device
        self.model.to(device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "neural"

    def act(self, obs, config=None) -> list:
        is_pointer = isinstance(self.model, PointerNetworkModel)

        if isinstance(obs, dict):
            player = obs.get("player", 0)
            raw_planets = obs.get("planets", [])
        else:
            player = obs.player
            raw_planets = list(obs.planets) if hasattr(obs.planets, "__iter__") else obs.planets

        is_planet_policy = isinstance(self.model, PlanetPolicyModel)

        if is_planet_policy:
            if isinstance(obs, dict):
                state = self.state_builder.from_obs(obs, player)
            else:
                state = self.state_builder.from_step(obs, player)
            pf = torch.tensor(state["planet_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
            ff = torch.tensor(state["fleet_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
            fm = torch.tensor(state["fleet_mask"], dtype=torch.bool).unsqueeze(0).to(self.device)
            gf = torch.tensor(state["global_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
            pm = torch.tensor(state["planet_mask"], dtype=torch.bool).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(pf, ff, fm, gf, pm)
            from .planet_policy_model import PlanetPolicyOutput
            squeezed = PlanetPolicyOutput(
                action_type_logits=output.action_type_logits.squeeze(0),
                target_logits=output.target_logits.squeeze(0),
                amount_logits=output.amount_logits.squeeze(0),
                value=output.value,
            )
            planets_arr = state["planet_features"]
            return self.codec.decode_per_planet(squeezed, state["context"], planets_arr, self.model.config.max_planets)

        if isinstance(obs, dict):
            if is_pointer:
                structured = self.state_builder.from_obs_structured(obs, player)
            else:
                model_input = self.state_builder.from_obs(obs, player)
        else:
            if is_pointer:
                structured = self.state_builder.from_step_structured(obs, player)
            else:
                model_input = self.state_builder.from_step(obs, player)

        with torch.no_grad():
            if is_pointer:
                pf = torch.from_numpy(structured["planet_features"]).unsqueeze(0).to(self.device)
                ff = torch.from_numpy(structured["fleet_features"]).unsqueeze(0).to(self.device)
                pm = torch.from_numpy(structured["planet_mask"]).unsqueeze(0).to(self.device)
                output = self.model(pf, ff, pm)
                context = structured["context"]
                output_single = PointerPolicyOutput(
                    action_type_logits=output.action_type_logits[0],
                    source_logits=output.source_logits[0],
                    target_logits=output.target_logits[0],
                    amount_logits=output.amount_logits[0],
                    value=output.value[0],
                )
            else:
                tensor = torch.from_numpy(model_input.array).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                context = model_input.context
                output_single = PolicyOutput(
                    action_type_logits=output.action_type_logits[0],
                    source_logits=output.source_logits[0],
                    target_logits=output.target_logits[0],
                    amount_logits=output.amount_logits[0],
                    value=output.value[0],
                )

        if len(raw_planets) > 0:
            planets_arr = np.array(raw_planets, dtype=np.float32)
        else:
            planets_arr = np.empty((0, 7), dtype=np.float32)

        return self.codec.decode(output_single, context, planets_arr)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NeuralBot":
        """Load a NeuralBot from a checkpoint file.

        Supports both flat (PolicyValueModel) and pointer (PointerNetworkModel) checkpoints.
        The checkpoint must contain a "model_type" key ("flat" or "pointer"); if absent,
        "flat" is assumed for backward compatibility.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model_type = checkpoint.get("model_type", "flat")

        if model_type == "planet_policy":
            config_dict = checkpoint["config"]
            config = PlanetPolicyConfig(
                Dp=config_dict["Dp"],
                Df=config_dict["Df"],
                Dg=config_dict["Dg"],
                E=config_dict["E"],
                F=config_dict["F"],
                G=config_dict["G"],
                max_planets=config_dict["max_planets"],
                max_fleets=config_dict["max_fleets"],
                n_amount_bins=config_dict["n_amount_bins"],
                dropout=config_dict["dropout"],
                n_attn_heads=config_dict["n_attn_heads"],
            )
            model = PlanetPolicyModel(config)
            model.load_state_dict(checkpoint["state_dict"])
            state_builder = StateBuilderV2(max_planets=config.max_planets, max_fleets=config.max_fleets)
            codec = ActionCodecV2(n_amount_bins=config.n_amount_bins)
            return cls(model=model, state_builder=state_builder, codec=codec, device=device)
        elif model_type == "pointer":
            config: PointerNetworkConfig = checkpoint["config"]
            model = PointerNetworkModel(config)
            model.load_state_dict(checkpoint["state_dict"])
            max_planets = config.max_planets
            max_fleets = config.max_fleets
            n_amount_bins = config.n_amount_bins
            state_builder = StateBuilder(max_planets=max_planets, max_fleets=max_fleets)
            codec = ActionCodec(n_amount_bins=n_amount_bins)
            return cls(model=model, state_builder=state_builder, codec=codec, device=device)
        else:
            config: PolicyValueConfig = checkpoint["config"]
            max_planets = checkpoint.get("max_planets", config.max_planets)
            max_fleets = checkpoint.get("max_fleets", 100)
            n_amount_bins = checkpoint.get("n_amount_bins", config.n_amount_bins)
            model = PolicyValueModel(config)
            model.load_state_dict(checkpoint["state_dict"])
            state_builder = StateBuilder(max_planets=max_planets, max_fleets=max_fleets)
            codec = ActionCodec(n_amount_bins=n_amount_bins)
            return cls(model=model, state_builder=state_builder, codec=codec, device=device)


# Module-level agent_fn for kaggle_environments compatibility
# Uses a default untrained model; replace with NeuralBot.load() after training.
_default_bot: NeuralBot | None = None


def agent_fn(obs, config=None):
    global _default_bot
    if _default_bot is None:
        _cfg = PolicyValueConfig(
            input_dim=StateBuilder().input_dim,
            hidden_dims=[256, 128],
        )
        _default_bot = NeuralBot(
            model=PolicyValueModel(_cfg),
            state_builder=StateBuilder(),
            codec=ActionCodec(),
        )
    return _default_bot.act(obs, config)
