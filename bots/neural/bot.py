"""NeuralBot — inference wrapper around PlanetPolicyModel."""

from __future__ import annotations

import torch

from bots.interface import Bot
from .planet_policy_model import PlanetPolicyConfig, PlanetPolicyModel, PlanetPolicyOutput
from .state_builder import StateBuilder
from .action_codec import ActionCodec


class NeuralBot(Bot):
    def __init__(
        self,
        model: PlanetPolicyModel,
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
        if not isinstance(self.model, PlanetPolicyModel):
            raise NotImplementedError(f"Unsupported model type: {type(self.model)}")

        if isinstance(obs, dict):
            player = obs.get("player", 0)
            state = self.state_builder.from_obs(obs, player)
        else:
            player = obs.player
            state = self.state_builder.from_step(obs, player)

        pf = torch.tensor(state["planet_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
        ff = torch.tensor(state["fleet_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
        fm = torch.tensor(state["fleet_mask"], dtype=torch.bool).unsqueeze(0).to(self.device)
        gf = torch.tensor(state["global_features"], dtype=torch.float32).unsqueeze(0).to(self.device)
        pm = torch.tensor(state["planet_mask"], dtype=torch.bool).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(pf, ff, fm, gf, pm)
        squeezed = PlanetPolicyOutput(
            action_type_logits=output.action_type_logits.squeeze(0),
            target_logits=output.target_logits.squeeze(0),
            amount_logits=output.amount_logits.squeeze(0),
            value=output.value,
        )
        planets_arr = state["planet_features"]
        return self.codec.decode_per_planet(squeezed, state["context"], planets_arr, self.model.config.max_planets)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NeuralBot":
        """Load a NeuralBot from a checkpoint file.

        Only planet_policy checkpoints are supported.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model_type = checkpoint.get("model_type", "planet_policy")

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
            state_builder = StateBuilder(max_planets=config.max_planets, max_fleets=config.max_fleets)
            codec = ActionCodec(n_amount_bins=config.n_amount_bins)
            return cls(model=model, state_builder=state_builder, codec=codec, device=device)
        else:
            raise ValueError(
                f"Unsupported model_type {model_type!r} in checkpoint {path!r}. "
                "Only 'planet_policy' checkpoints are supported."
            )


# Module-level agent_fn for kaggle_environments compatibility
# Uses a default untrained model; replace with NeuralBot.load() after training.
_default_bot: NeuralBot | None = None


def agent_fn(obs, config=None):
    global _default_bot
    if _default_bot is None:
        _cfg = PlanetPolicyConfig()
        _default_bot = NeuralBot(
            model=PlanetPolicyModel(_cfg),
            state_builder=StateBuilder(max_planets=_cfg.max_planets, max_fleets=_cfg.max_fleets),
            codec=ActionCodec(n_amount_bins=_cfg.n_amount_bins),
        )
    return _default_bot.act(obs, config)
