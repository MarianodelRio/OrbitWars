"""ILTrainer: imitation learning training loop."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _safe_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -1,
    ce_override: nn.CrossEntropyLoss | None = None,
) -> torch.Tensor:
    """CrossEntropyLoss that returns 0 (not NaN) when all labels are ignored.

    ce_override: if provided, use this loss fn (e.g. with class weights) instead of
    bare F.cross_entropy. Weight tensor must already be on the correct device.
    """
    mask = labels != ignore_index
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)
    if ce_override is not None:
        ce_override = ce_override.to(logits.device)
        return ce_override(logits[mask], labels[mask])
    return F.cross_entropy(logits[mask], labels[mask])

from pathlib import Path

from dataset.catalog import DataCatalog
from bots.neural.training import (
    build_il_dataset,
    build_il_cache,
    load_precomputed_split,
    NeuralILDataset,
    PrecomputedILDataset,
)
from bots.neural.planet_policy_model import PlanetPolicyModel
from training.trainers.base_trainer import BaseTrainer


class ILTrainer(BaseTrainer):
    def train(self) -> None:
        # 1. Setup run dir, loggers, checkpoint manager
        self._setup_run_dir()

        # 2. Build full catalog from config.data_pipeline
        pipeline = self.config.data_pipeline
        catalog_cfg = pipeline.get("catalog", {})
        builder_cfg = pipeline.get("builder", {})

        roots = catalog_cfg.get("roots", None)
        if roots is not None:
            roots = [Path(r) for r in roots]

        catalog = DataCatalog.scan(roots=roots)

        # Apply filters from catalog config
        filter_cfg = catalog_cfg.get("filter", {})
        if any(v is not None for v in filter_cfg.values()):
            catalog = catalog.filter(
                bot=filter_cfg.get("bot"),
                opponent=filter_cfg.get("opponent"),
                winner_only=filter_cfg.get("winner_only", False),
                done_reason=filter_cfg.get("done_reason"),
                min_steps=filter_cfg.get("min_steps"),
                max_steps=filter_cfg.get("max_steps"),
            )
        max_episodes = filter_cfg.get("max_episodes")
        if max_episodes is not None:
            catalog = DataCatalog(catalog.episodes[:max_episodes])

        # 3. Split by episode — shuffle first so each split has a mix of matchups
        import random as _random
        all_episodes = list(catalog.episodes)
        rng = _random.Random(self.config.seed)
        rng.shuffle(all_episodes)
        n_val = max(1, int(len(all_episodes) * self.config.val_split))

        train_episodes = all_episodes[:-n_val]
        val_episodes = all_episodes[-n_val:]

        train_catalog = DataCatalog(train_episodes)
        val_catalog = DataCatalog(val_episodes)

        perspective = builder_cfg.get("perspective", "winner")

        # Resolve step_filter from config string
        step_filter = None
        step_filter_str = builder_cfg.get("step_filter")
        if step_filter_str == "non_empty_state":
            from dataset.transforms.filters import NonEmptyStateFilter
            step_filter = NonEmptyStateFilter()
        elif step_filter_str is not None:
            print(f"[ILTrainer] Warning: unknown step_filter {step_filter_str!r}, ignoring")

        # -----------------------------------------------------------------
        # Dataset: use pre-computed HDF5 cache when cache_path is configured.
        # Falls back to lazy NeuralILDataset if cache_path is absent.
        # -----------------------------------------------------------------
        cache_path_str = builder_cfg.get("cache_path")
        use_cache = cache_path_str is not None

        if use_cache:
            cache_path = Path(cache_path_str)
            if not cache_path.exists():
                print(f"[ILTrainer] Building IL cache → {cache_path}")
                print(f"            (one-time cost; subsequent runs will be fast)")
                build_il_cache(
                    catalog,
                    self.state_builder,
                    self.codec,
                    cache_path,
                    step_filter=step_filter,
                    perspective=perspective,
                )
            else:
                print(f"[ILTrainer] Loading IL cache from {cache_path}")

            train_dataset, val_dataset = load_precomputed_split(
                cache_path,
                catalog,
                train_episodes,
                val_episodes,
            )
        else:
            train_dataset = build_il_dataset(
                train_catalog, self.state_builder, self.codec,
                perspective=perspective,
                step_filter=step_filter,
            )
            val_dataset = build_il_dataset(
                val_catalog, self.state_builder, self.codec,
                perspective=perspective,
                step_filter=step_filter,
            )

        # shuffle=False: index is episode-ordered so the LRU reader cache in
        # NeuralILDataset achieves near-100% hit rate.  Episode-level shuffling
        # already happens above (rng.shuffle(all_episodes)).
        # For PrecomputedILDataset we also keep shuffle=False: samples are
        # ordered by episode, so HDF5 reads are sequential (cache-friendly).
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Summary
        total_steps = sum(e.total_steps for e in all_episodes)
        bots_seen = sorted({e.bot0 for e in all_episodes} | {e.bot1 for e in all_episodes})
        print("\n=== Data ===")
        print(f"  Episodes : {len(all_episodes)}  (train={len(train_episodes)}, val={len(val_episodes)})")
        print(f"  Turns    : {total_steps:,}")
        print(f"  Bots     : {', '.join(bots_seen)}")
        print(f"  Samples  : train={len(train_dataset):,}  val={len(val_dataset):,}")
        print(f"  Batches  : {len(train_loader)} per epoch  (batch_size={self.config.batch_size})")
        cfg = self.model.config
        total_params = sum(p.numel() for p in self.model.parameters())
        weight_decay = self.config.weight_decay
        action_type_loss_weight = self.config.action_type_loss_weight
        value_loss_weight = self.config.value_loss_weight
        use_class_weights = self.config.use_class_weights
        print("\n=== Model ===")
        print(f"  Type         : PlanetPolicy")
        print(f"  Architecture : planet_encoder({cfg.Dp}→{cfg.E}) fleet_encoder({cfg.Df}→{cfg.F}) global({cfg.G}) n_attn_heads={cfg.n_attn_heads}")
        print(f"  Parameters   : {total_params:,}")
        print(f"  Dropout      : {cfg.dropout}  |  lr={self.config.lr}  |  wd={weight_decay}  |  epochs={self.config.epochs}")
        print(f"  Perspective  : {perspective}  |  device={self.config.device}")
        print(f"  Loss weights : action_type={action_type_loss_weight}  value={value_loss_weight}  class_weights={use_class_weights}")
        print()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=weight_decay)

        # LR scheduler
        _lr_schedule = getattr(self.config, "lr_schedule", "constant")
        if _lr_schedule == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            _lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs, eta_min=self.config.lr * 0.01)
        elif _lr_schedule == "step":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            _lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        else:
            _lr_scheduler = None

        # 5. Loss functions — optionally weighted by inverse class frequency
        mse_loss = nn.MSELoss()
        if use_class_weights:
            import numpy as _np
            if use_cache and isinstance(train_dataset, PrecomputedILDataset):
                # Fast path: class weight counts pre-computed during cache build.
                # Note: counts are from the full dataset (all episodes), which is a
                # good proxy for the train split (85% of data with same distribution).
                at_counts, amt_counts = train_dataset.class_weight_counts
            else:
                # Slow fallback: iterate train samples (used only without cache)
                at_labels = [
                    v.item()
                    for i in range(len(train_dataset))
                    for v in train_dataset[i]["action_types"]
                    if v.item() != -1
                ]
                amt_labels_raw = [
                    v.item()
                    for i in range(len(train_dataset))
                    for v in train_dataset[i]["amount_bins"]
                    if v.item() != -1
                ]
                at_counts = _np.bincount(at_labels, minlength=2).astype(_np.float32)
                amt_valid = [a for a in amt_labels_raw if a >= 0]
                amt_counts = _np.bincount(amt_valid, minlength=5).astype(_np.float32)

            at_counts = _np.clip(at_counts, 1, None)
            at_weights = 1.0 / at_counts
            at_weights = _np.clip(at_weights / at_weights.mean(), 0.1, 10.0)

            amt_counts = _np.clip(amt_counts, 1, None)
            amt_weights = 1.0 / amt_counts
            amt_weights = _np.clip(amt_weights / amt_weights.mean(), 0.1, 10.0)

            print(f"  Class weights action_type : {[round(w, 3) for w in at_weights.tolist()]}")
            print(f"  Class weights amount_bin  : {[round(w, 3) for w in amt_weights.tolist()]}")
            print()

            ce_action_type = nn.CrossEntropyLoss(weight=torch.tensor(at_weights))
            ce_amount = nn.CrossEntropyLoss(weight=torch.tensor(amt_weights))
        else:
            ce_action_type = nn.CrossEntropyLoss()
            ce_amount = nn.CrossEntropyLoss()

        device = torch.device(self.config.device)
        self.model.to(device)
        self.model.train()

        best_val_loss = float("inf")
        start_epoch = 1
        epochs_no_improve = 0

        # Resume from checkpoint if configured
        resume_from = getattr(self.config, "resume_from", None)
        if resume_from is not None:
            ckpt_path = Path(resume_from)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device)
                # Support both key names used by CheckpointManager
                model_weights = ckpt.get("model_state_dict") or ckpt.get("state_dict")
                if model_weights is not None:
                    self.model.load_state_dict(model_weights)
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_loss = ckpt.get("val_loss") or ckpt.get("best_val_loss") or float("inf")
                if best_val_loss is None:
                    best_val_loss = float("inf")
                print(f"[ILTrainer] Resumed from {ckpt_path}  (next epoch={start_epoch}, best_val_loss={best_val_loss:.4f})")
            else:
                print(f"[ILTrainer] Warning: resume_from={resume_from!r} not found, starting fresh")

        # Try importing evaluator; graceful fallback if not yet available
        try:
            from training.evaluation.evaluator import Evaluator
            _evaluator_available = True
        except ImportError:
            _evaluator_available = False
            print("[ILTrainer] Warning: training.evaluation.evaluator not available; evaluation will be skipped.")

        # 6. Epoch loop
        for epoch in range(start_epoch, self.config.epochs + 1):
            # a. Train epoch
            self.model.train()
            train_loss_total = 0.0
            train_steps = 0

            for batch in train_loader:
                optimizer.zero_grad()
                output = self.model(
                    batch["planet_features"].to(device),
                    batch["fleet_features"].to(device),
                    batch["fleet_mask"].to(device),
                    batch["global_features"].to(device),
                    batch["planet_mask"].to(device),
                )
                B, N = batch["action_types"].shape
                assert N == self.model.config.max_planets, f"Batch planet dim {N} != model max_planets {self.model.config.max_planets}"
                at_flat = batch["action_types"].to(device).view(B * N)
                tgt_flat = batch["target_idxs"].to(device).view(B * N)
                amt_flat = batch["amount_bins"].to(device).view(B * N)
                at_logits = output.action_type_logits.view(B * N, 2)
                tgt_logits = output.target_logits.view(B * N, N)
                amt_logits = output.amount_logits.view(B * N, self.model.config.n_amount_bins)
                value_labels = batch["value_target"].to(device).float()
                loss = (
                    action_type_loss_weight * _safe_ce(at_logits, at_flat, ce_override=ce_action_type)
                    + _safe_ce(tgt_logits, tgt_flat)
                    + _safe_ce(amt_logits, amt_flat, ce_override=ce_amount)
                    + value_loss_weight * mse_loss(output.value.squeeze(-1).squeeze(-1), value_labels)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_total += loss.item()
                train_steps += 1

            train_loss = train_loss_total / max(train_steps, 1)

            # b. Val epoch
            self.model.eval()
            val_loss_total = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    output = self.model(
                        batch["planet_features"].to(device),
                        batch["fleet_features"].to(device),
                        batch["fleet_mask"].to(device),
                        batch["global_features"].to(device),
                        batch["planet_mask"].to(device),
                    )
                    B, N = batch["action_types"].shape
                    assert N == self.model.config.max_planets, f"Batch planet dim {N} != model max_planets {self.model.config.max_planets}"
                    at_flat = batch["action_types"].to(device).view(B * N)
                    tgt_flat = batch["target_idxs"].to(device).view(B * N)
                    amt_flat = batch["amount_bins"].to(device).view(B * N)
                    at_logits = output.action_type_logits.view(B * N, 2)
                    tgt_logits = output.target_logits.view(B * N, N)
                    amt_logits = output.amount_logits.view(B * N, self.model.config.n_amount_bins)
                    value_labels = batch["value_target"].to(device).float()
                    loss = (
                        action_type_loss_weight * _safe_ce(at_logits, at_flat, ce_override=ce_action_type)
                        + _safe_ce(tgt_logits, tgt_flat)
                        + _safe_ce(amt_logits, amt_flat, ce_override=ce_amount)
                        + value_loss_weight * mse_loss(output.value.squeeze(-1).squeeze(-1), value_labels)
                    )

                    val_loss_total += loss.item()
                    val_steps += 1

            val_loss = val_loss_total / max(val_steps, 1)

            # Step LR scheduler
            if _lr_scheduler is not None:
                if isinstance(_lr_scheduler, __import__("torch").optim.lr_scheduler.ReduceLROnPlateau):
                    _lr_scheduler.step(val_loss)
                else:
                    _lr_scheduler.step()

            # c. Log
            self._log_train({"epoch": epoch, "loss": train_loss})
            self._log_val({"epoch": epoch, "loss": val_loss})

            print(f"[Epoch {epoch}/{self.config.epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            # e. Check best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            _patience = getattr(self.config, "early_stopping_patience", 0)
            if _patience > 0 and epochs_no_improve >= _patience:
                print(f"[ILTrainer] Early stopping at epoch {epoch} (no improvement for {_patience} epochs)")
                self._save_checkpoint(
                    epoch,
                    {"train_loss": train_loss, "val_loss": val_loss},
                    is_best,
                )
                break

            # f. Save checkpoint
            self._save_checkpoint(
                epoch,
                {"train_loss": train_loss, "val_loss": val_loss},
                is_best,
            )

            # g. Evaluation
            if _evaluator_available and epoch % self.config.eval_every == 0:
                evaluator = Evaluator(
                    bot=self._ckpt_manager.load_bot(tag="last", device=self.config.device),
                    opponents=self.config.eval_opponents,
                    n_matches=self.config.n_eval_matches,
                    run_dir=self.config.run_dir,
                )
                results = evaluator.run(epoch=epoch)
                for opp, res in results.items():
                    print(f"  [Eval vs {opp}] win_rate={res.get('win_rate'):.2f}")
