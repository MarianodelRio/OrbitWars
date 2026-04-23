"""ILTrainer: imitation learning training loop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.catalog import DataCatalog
from bots.neural.training import build_il_dataset, NeuralILDataset
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
            from pathlib import Path
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

        # 3. Split by episode index to avoid data leakage
        all_episodes = catalog.episodes
        n_val = max(1, int(len(all_episodes) * self.config.val_split))

        train_episodes = all_episodes[:-n_val]
        val_episodes = all_episodes[-n_val:]

        train_catalog = DataCatalog(train_episodes)
        val_catalog = DataCatalog(val_episodes)

        perspective = builder_cfg.get("perspective", "winner")

        train_dataset = build_il_dataset(
            train_catalog, self.state_builder, self.codec,
            perspective=perspective,
        )
        val_dataset = build_il_dataset(
            val_catalog, self.state_builder, self.codec,
            perspective=perspective,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # 4. Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # 5. Loss functions
        ce_loss = nn.CrossEntropyLoss()
        ce_ignore = nn.CrossEntropyLoss(ignore_index=-1)
        mse_loss = nn.MSELoss()

        device = torch.device(self.config.device)
        self.model.to(device)
        self.model.train()

        best_val_loss = float("inf")

        # Try importing evaluator; graceful fallback if not yet available
        try:
            from training.evaluation.evaluator import Evaluator
            _evaluator_available = True
        except ImportError:
            _evaluator_available = False
            print("[ILTrainer] Warning: training.evaluation.evaluator not available; evaluation will be skipped.")

        # 6. Epoch loop
        for epoch in range(1, self.config.epochs + 1):
            # a. Train epoch
            self.model.train()
            train_loss_total = 0.0
            train_steps = 0

            for batch in train_loader:
                states = batch["state"].to(device)
                action_type_labels = batch["action_type"].to(device)
                source_labels = batch["source_idx"].to(device)
                target_labels = batch["target_idx"].to(device)
                amount_labels = batch["amount_bin"].to(device)
                value_labels = batch["value_target"].to(device)

                optimizer.zero_grad()
                output = self.model(states)

                loss = (
                    ce_loss(output.action_type_logits, action_type_labels)
                    + ce_ignore(output.source_logits, source_labels)
                    + ce_ignore(output.target_logits, target_labels)
                    + ce_ignore(output.amount_logits, amount_labels)
                    + 0.5 * mse_loss(output.value.squeeze(-1), value_labels)
                )

                loss.backward()
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
                    states = batch["state"].to(device)
                    action_type_labels = batch["action_type"].to(device)
                    source_labels = batch["source_idx"].to(device)
                    target_labels = batch["target_idx"].to(device)
                    amount_labels = batch["amount_bin"].to(device)
                    value_labels = batch["value_target"].to(device)

                    output = self.model(states)

                    loss = (
                        ce_loss(output.action_type_logits, action_type_labels)
                        + ce_ignore(output.source_logits, source_labels)
                        + ce_ignore(output.target_logits, target_labels)
                        + ce_ignore(output.amount_logits, amount_labels)
                        + 0.5 * mse_loss(output.value.squeeze(-1), value_labels)
                    )

                    val_loss_total += loss.item()
                    val_steps += 1

            val_loss = val_loss_total / max(val_steps, 1)

            # c. Log
            self._log_train({"epoch": epoch, "loss": train_loss})
            self._log_val({"epoch": epoch, "loss": val_loss})

            print(f"[Epoch {epoch}/{self.config.epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            # e. Check best
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

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
