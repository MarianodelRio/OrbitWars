"""NeuralBot — inference wrapper around PolicyValueModel."""

from __future__ import annotations

import numpy as np
import torch

from bots.interface import Bot
from .action_codec import ActionCodec
from .model import PolicyOutput, PolicyValueConfig, PolicyValueModel
from .state_builder import StateBuilder


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
        if isinstance(obs, dict):
            player = obs.get("player", 0)
            raw_planets = obs.get("planets", [])
            model_input = self.state_builder.from_obs(obs, player)
        else:
            player = obs.player
            raw_planets = list(obs.planets) if hasattr(obs.planets, "__iter__") else obs.planets
            model_input = self.state_builder.from_step(obs, player)

        tensor = torch.from_numpy(model_input.array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

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

        return self.codec.decode(output_single, model_input.context, planets_arr)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NeuralBot":
        """Load a NeuralBot from a checkpoint file.

        Checkpoint format:
            {
                "config": PolicyValueConfig,
                "state_dict": dict,
                "max_planets": int,
                "max_fleets": int,
                "n_amount_bins": int,
            }
        """
        checkpoint = torch.load(path, map_location=device)
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
