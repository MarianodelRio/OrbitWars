"""PipelineConfig — JSON-driven factory for DataCatalog and SampleBuilder."""

from __future__ import annotations

import json
from pathlib import Path

from dataset.catalog import DataCatalog
from dataset.builder import SampleBuilder
from dataset.transforms.state import RawStateTransform
from dataset.transforms.action import RawActionTransform
from dataset.transforms.filters import HasActionFilter, EarlyGameFilter, NonEmptyStateFilter
from dataset.transforms.reward import BinaryOutcomeReward


class PipelineConfig:
    def __init__(self, catalog_cfg: dict, builder_cfg: dict) -> None:
        self._catalog_cfg = catalog_cfg
        self._builder_cfg = builder_cfg

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw = json.load(f)
        return cls(
            catalog_cfg=raw["catalog"],
            builder_cfg=raw["builder"],
        )

    def build_catalog(self) -> DataCatalog:
        cfg = self._catalog_cfg

        # Resolve roots
        roots_raw = cfg.get("roots")
        if roots_raw is not None:
            roots = [Path(r) for r in roots_raw]
        else:
            roots = None

        catalog = DataCatalog.scan(roots=roots)

        # Apply filters — only pass keys with non-None / non-False values
        filter_cfg = cfg.get("filter", {}) or {}

        # Guard: winner_only=True requires bot to be specified
        if filter_cfg.get("winner_only") is True and filter_cfg.get("bot") is None:
            raise ValueError(
                "pipeline.json has winner_only=true but bot is null. "
                "winner_only requires a bot name to be specified."
            )

        kwargs = {}
        for key in ("bot", "opponent", "done_reason", "min_steps", "max_steps"):
            val = filter_cfg.get(key)
            if val is not None:
                kwargs[key] = val

        winner_only = filter_cfg.get("winner_only", False)
        if winner_only:
            kwargs["winner_only"] = True

        if kwargs:
            catalog = catalog.filter(**kwargs)

        return catalog

    def build_builder(self) -> SampleBuilder:
        cfg = self._builder_cfg
        return SampleBuilder(
            state_transform=self._resolve_state_transform(cfg["state_transform"]),
            action_transform=self._resolve_action_transform(cfg["action_transform"]),
            reward_transform=self._resolve_reward_transform(cfg.get("reward_transform")),
            step_filter=self._resolve_step_filter(cfg.get("step_filter")),
            perspective=cfg["perspective"],
            mode=cfg["mode"],
        )

    # ------------------------------------------------------------------
    # Internal resolvers
    # ------------------------------------------------------------------

    def _resolve_state_transform(self, name: str):
        if name == "raw":
            return RawStateTransform()
        raise ValueError(
            f"Unknown state_transform {name!r}. Expected one of: 'raw'."
        )

    def _resolve_action_transform(self, name: str):
        if name == "raw":
            return RawActionTransform()
        raise ValueError(
            f"Unknown action_transform {name!r}. Expected one of: 'raw'."
        )

    def _resolve_reward_transform(self, name):
        if name is None:
            return None
        if name == "binary_outcome":
            return BinaryOutcomeReward()
        raise ValueError(
            f"Unknown reward_transform {name!r}. Expected one of: null, 'binary_outcome'."
        )

    def _resolve_step_filter(self, spec):
        if spec is None:
            return None
        if spec == "has_action":
            return HasActionFilter()
        if spec == "non_empty_state":
            return NonEmptyStateFilter()
        if isinstance(spec, str) and spec.startswith("early_game:"):
            parts = spec.split(":")
            if len(parts) != 2 or not parts[1]:
                raise ValueError(
                    f"Malformed step_filter {spec!r}. Expected format: 'early_game:<N>' "
                    "where N is an integer (e.g. 'early_game:200')."
                )
            try:
                max_turn = int(parts[1])
            except ValueError:
                raise ValueError(
                    f"Malformed step_filter {spec!r}. The N in 'early_game:<N>' must be "
                    f"an integer, got {parts[1]!r}."
                )
            return EarlyGameFilter(max_turn=max_turn)
        raise ValueError(
            f"Unknown step_filter {spec!r}. Expected one of: null, 'has_action', "
            "'non_empty_state', 'early_game:<N>'."
        )
