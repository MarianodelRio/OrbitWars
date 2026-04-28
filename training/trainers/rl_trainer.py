"""RLTrainer: PPO-based reinforcement learning trainer for Orbit Wars."""

from __future__ import annotations

import dataclasses

import torch

from training.utils.rl_config import RLConfig
from training.utils.checkpointing import CheckpointManager
from training.utils.rl_metrics import RLMetricsLogger
from training.rewards.potential import PotentialReward
from training.envs.orbit_env import OrbitWarsEnv
from training.rl.rollout_buffer import RolloutBuffer, RolloutStep
from training.rl.opponent_pool import OpponentPool
from training.rl.ppo import compute_ppo_loss, PPOLossResult
from bots.neural.planet_policy_model import PlanetPolicyModel, PlanetPolicyOutput
from bots.neural.policy_sampler import PolicySampler
from training.trainers.il_trainer import _safe_ce  # used for IL distillation minibatches


def _compute_kl_bc_coef(iteration: int, cfg) -> float:
    if not cfg.bc_policy_path:
        return 0.0
    if iteration <= cfg.kl_bc_coef_decay_iters:
        t = (iteration - 1) / max(cfg.kl_bc_coef_decay_iters - 1, 1)
        return cfg.kl_bc_coef_start + t * (cfg.kl_bc_coef_end - cfg.kl_bc_coef_start)
    return cfg.kl_bc_coef_end


class RLTrainer:
    def __init__(
        self,
        config: RLConfig,
        model: PlanetPolicyModel,
        state_builder,
        codec,
    ) -> None:
        self.config = config
        self.model = model
        self.state_builder = state_builder
        self.codec = codec

        # Initialized lazily in _setup()
        self._ckpt_manager: CheckpointManager | None = None
        self._rl_metrics_logger: RLMetricsLogger | None = None
        self._optimizer = None
        self._lr_scheduler = None
        self._env: OrbitWarsEnv | None = None
        self._pool: OpponentPool | None = None
        self._sampler: PolicySampler | None = None
        self._reward_fn: PotentialReward | None = None
        self._buffer: RolloutBuffer | None = None
        self._bc_model = None
        self._il_loader = None
        self._il_iter = None

    def _setup(self) -> None:
        cfg = self.config
        run_dir = cfg.run_dir
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / "eval").mkdir(parents=True, exist_ok=True)
        cfg.save(run_dir)

        self._ckpt_manager = CheckpointManager(run_dir)
        self._rl_metrics_logger = RLMetricsLogger(run_dir)

        self._optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        if cfg.lr_schedule == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self._lr_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=cfg.total_iterations,
                eta_min=cfg.lr * 0.01,
            )
        else:
            self._lr_scheduler = None

        self._reward_fn = PotentialReward(
            w_planets=cfg.w_planets,
            w_production=cfg.w_production,
            w_ships=cfg.w_ships,
            gamma=cfg.gamma,
            lam=cfg.reward_lambda,
            clip_abs=cfg.reward_clip_abs,  # deprecated but kept for compat
            r_terminal_win=cfg.r_terminal_win,
            r_terminal_loss=cfg.r_terminal_loss,
            r_terminal_margin_coef=cfg.r_terminal_margin_coef,
            r_event_capture_enemy=cfg.r_event_capture_enemy,
            r_event_capture_comet=cfg.r_event_capture_comet,
            r_event_eliminate_opponent=cfg.r_event_eliminate_opponent,
            r_event_lose_planet=cfg.r_event_lose_planet,
            r_event_ships_wasted_coef=cfg.r_event_ships_wasted_coef,
            r_explore=cfg.r_explore,
            explore_iterations=cfg.explore_iterations,
        )

        self._env = OrbitWarsEnv(self.state_builder, self._reward_fn, cfg.steps_per_episode)

        self._pool = OpponentPool(max_snapshots=cfg.max_snapshots)
        for opp_path in cfg.heuristic_opponents:
            name = opp_path.split(":")[-1]
            self._pool.add_heuristic(name=name, agent_fn_path=opp_path)
        if cfg.frozen_checkpoint:
            self._pool.add_frozen_checkpoint(cfg.frozen_checkpoint)

        self._sampler = PolicySampler(
            bins=self.codec.BINS,
            max_planets=self.model.config.max_planets,
        )

        self._buffer = RolloutBuffer(capacity=cfg.n_rollout_steps)

        # Load frozen BC reference policy for KL regularization
        if cfg.bc_policy_path:
            from pathlib import Path
            bc_path = Path(cfg.bc_policy_path)
            if bc_path.exists():
                try:
                    from bots.neural.planet_policy_model import PlanetPolicyModel
                    ckpt = torch.load(str(bc_path), map_location=cfg.device)
                    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
                    self._bc_model = PlanetPolicyModel(self.model.config)
                    self._bc_model.load_state_dict(state_dict, strict=False)
                    self._bc_model.eval()
                    self._bc_model.to(cfg.device)
                    for p in self._bc_model.parameters():
                        p.requires_grad_(False)
                    print(f"[RLTrainer] Loaded BC policy from {bc_path}")
                except Exception as e:
                    print(f"[RLTrainer] WARNING: failed to load BC policy: {e}")
                    self._bc_model = None
            else:
                print(f"[RLTrainer] WARNING: bc_policy_path '{bc_path}' not found; KL term disabled")

        # IL distillation DataLoader setup
        if cfg.il_distill_ratio > 0.0 and cfg.il_data_cache_path:
            from pathlib import Path
            il_path = Path(cfg.il_data_cache_path)
            if il_path.exists():
                try:
                    from bots.neural.training import PrecomputedILDataset
                    from torch.utils.data import DataLoader
                    il_dataset = PrecomputedILDataset(il_path)
                    self._il_loader = DataLoader(
                        il_dataset,
                        batch_size=cfg.ppo_batch_size,
                        shuffle=True,
                        num_workers=0,
                        drop_last=True,
                    )
                    self._il_iter = iter(self._il_loader)
                    print(f"[RLTrainer] IL distillation enabled, ratio={cfg.il_distill_ratio}")
                except Exception as e:
                    print(f"[RLTrainer] WARNING: failed to load IL dataset: {e}")
                    self._il_loader = None

        torch.manual_seed(cfg.seed)
        self._best_mean_winrate = 0.0

    def _collect_rollout(self) -> None:
        cfg = self.config
        device = cfg.device
        buffer = self._buffer
        env = self._env
        sampler = self._sampler

        opponent_fn = self._pool.sample(
            self_play_prob=self.config.self_play_prob,
            current_model_fn=self._current_model_as_agent(),
        )
        env.set_opponent(opponent_fn)
        state, info = env.reset()
        hidden = None

        while not buffer.is_full():
            pf = torch.tensor(state["planet_features"], dtype=torch.float32).unsqueeze(0).to(device)
            ff = torch.tensor(state["fleet_features"], dtype=torch.float32).unsqueeze(0).to(device)
            fm = torch.tensor(state["fleet_mask"], dtype=torch.bool).unsqueeze(0).to(device)
            gf = torch.tensor(state["global_features"], dtype=torch.float32).unsqueeze(0).to(device)
            pm = torch.tensor(state["planet_mask"], dtype=torch.bool).unsqueeze(0).to(device)

            rl_masks = sampler.build_masks(state["context"], device=device)

            with torch.no_grad():
                # relational_tensor not passed: rel_proj was never trained during IL
                # (not in HDF5 cache), so its random weights cause NaN. Pass None
                # to use the same code path as IL. (tracked as technical debt — fix: Option B)
                output, new_hidden = self.model(pf, ff, fm, gf, pm, None, hidden)
                output_squeezed = PlanetPolicyOutput(
                    action_type_logits=output.action_type_logits.squeeze(0),
                    target_logits=output.target_logits.squeeze(0),
                    amount_logits=output.amount_logits.squeeze(0),
                    v_outcome=output.v_outcome.squeeze(0),
                    v_score_diff=output.v_score_diff.squeeze(0),
                    v_shaped=output.v_shaped.squeeze(0),
                )
                sample_result = sampler.sample(
                    output_squeezed,
                    rl_masks,
                    state["context"],
                    state["planet_features"],
                    deterministic=False,
                )

            next_state, reward, done, step_info = env.step(sample_result.game_actions)

            if step_info.get("error") and step_info.get("error_reason") == "env_error":
                # Reset and continue
                opponent_fn = self._pool.sample(
                    self_play_prob=self.config.self_play_prob,
                    current_model_fn=self._current_model_as_agent(),
                )
                env.set_opponent(opponent_fn)
                state, _ = env.reset()
                hidden = None
                continue

            roll_step = RolloutStep(
                state=dict(state),
                rl_masks=rl_masks,
                canonical=sample_result.canonical,
                log_prob_old=sample_result.log_prob.item(),
                value=sample_result.value.item(),
                reward=reward,
                done=done,
                terminal_reward=step_info.get("terminal_reward", 0.0),
                shaped_reward=step_info.get("shaped_reward", 0.0),
                player=self._env._player,
                step_count=step_info.get("step", 0),
                h_n=hidden[0].detach().cpu() if hidden is not None else None,
                c_n=hidden[1].detach().cpu() if hidden is not None else None,
            )
            buffer.add(roll_step)
            hidden = new_hidden

            if done:
                opponent_fn = self._pool.sample(
                    self_play_prob=self.config.self_play_prob,
                    current_model_fn=self._current_model_as_agent(),
                )
                env.set_opponent(opponent_fn)
                state, _ = env.reset()
                hidden = None
            else:
                state = next_state

        # Bootstrap value for last step
        if self._buffer._steps and self._buffer._steps[-1].done:
            last_value = 0.0
        else:
            pf = torch.tensor(state["planet_features"], dtype=torch.float32).unsqueeze(0).to(device)
            ff = torch.tensor(state["fleet_features"], dtype=torch.float32).unsqueeze(0).to(device)
            fm = torch.tensor(state["fleet_mask"], dtype=torch.bool).unsqueeze(0).to(device)
            gf = torch.tensor(state["global_features"], dtype=torch.float32).unsqueeze(0).to(device)
            pm = torch.tensor(state["planet_mask"], dtype=torch.bool).unsqueeze(0).to(device)
            with torch.no_grad():
                output, _ = self.model(pf, ff, fm, gf, pm, None, hidden)
                last_value = output.v_shaped.squeeze().item()

        self._buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

    def _ppo_update(self, iteration: int = 1, kl_bc_coef: float = 0.0) -> PPOLossResult:
        import random
        cfg = self.config
        device = cfg.device
        all_results = []
        for _epoch in range(cfg.ppo_epochs):
            batches = self._buffer.get_batches(cfg.ppo_batch_size, cfg.device)
            for batch in batches:
                # IL distillation: replace ~il_distill_ratio fraction of minibatches with CE batches
                if self._il_loader is not None and random.random() < cfg.il_distill_ratio:
                    try:
                        il_batch = next(self._il_iter)
                    except StopIteration:
                        self._il_iter = iter(self._il_loader)
                        il_batch = next(self._il_iter)
                    # Move to device
                    il_pf = il_batch["planet_features"].to(device)
                    il_ff = il_batch["fleet_features"].to(device)
                    il_fm = il_batch["fleet_mask"].to(device)
                    il_gf = il_batch["global_features"].to(device)
                    il_pm = il_batch["planet_mask"].to(device)
                    il_at = il_batch["action_types"].to(device)
                    il_ti = il_batch["target_idxs"].to(device)
                    il_ab = il_batch["amount_bins"].to(device)

                    self.model.train()
                    il_output, _ = self.model(il_pf, il_ff, il_fm, il_gf, il_pm)
                    # CE losses (flat, ignore_index=-1)
                    B_il, P_il = il_at.shape
                    il_loss = (
                        _safe_ce(il_output.action_type_logits.view(B_il * P_il, -1), il_at.view(-1))
                        + _safe_ce(il_output.target_logits.view(B_il * P_il, -1), il_ti.view(-1))
                        + _safe_ce(il_output.amount_logits.view(B_il * P_il, -1), il_ab.view(-1))
                    )
                    self._optimizer.zero_grad()
                    il_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self._optimizer.step()
                    # Skip PPO loss for this minibatch
                    continue

                self._optimizer.zero_grad()
                loss, result = compute_ppo_loss(
                    self.model, batch, cfg,
                    bc_model=self._bc_model,
                    kl_bc_coef=kl_bc_coef,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self._optimizer.step()
                all_results.append(result)

        if not all_results:
            return PPOLossResult(
                total_loss=0.0,
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                approx_kl=0.0,
                clip_fraction=0.0,
                explained_variance=0.0,
                kl_bc=0.0,
            )

        n = len(all_results)
        avg = PPOLossResult(
            total_loss=sum(r.total_loss for r in all_results) / n,
            policy_loss=sum(r.policy_loss for r in all_results) / n,
            value_loss=sum(r.value_loss for r in all_results) / n,
            entropy=sum(r.entropy for r in all_results) / n,
            approx_kl=sum(r.approx_kl for r in all_results) / n,
            clip_fraction=sum(r.clip_fraction for r in all_results) / n,
            explained_variance=sum(r.explained_variance for r in all_results) / n,
            kl_bc=sum(r.kl_bc for r in all_results) / n,
        )
        return avg

    def _current_model_as_agent(self):
        """Return a callable that wraps the current model as agent, restoring train mode."""
        from bots.neural.bot import NeuralBot
        from bots.interface import make_agent
        was_training = self.model.training
        bot = NeuralBot(
            model=self.model,
            state_builder=self.state_builder,
            codec=self.codec,
            device=self.config.device,
        )
        if was_training:
            self.model.train()
        return make_agent(bot)

    def train(self) -> None:
        self._setup()
        cfg = self.config
        self.model.to(cfg.device)

        # Auto-resume from last RL checkpoint if available
        start_iter = 1
        rl_last_path = cfg.run_dir / "checkpoints" / "rl_last.pt"
        if rl_last_path.exists():
            try:
                ckpt = self._ckpt_manager.load_rl_checkpoint(tag="rl_last", device=cfg.device)
                self.model.load_state_dict(ckpt["state_dict"])
                self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if self._lr_scheduler is not None and ckpt.get("lr_scheduler_state_dict") is not None:
                    self._lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
                start_iter = ckpt.get("iteration", 0) + 1
                print(f"[RLTrainer] Resumed from rl_last.pt  (next iteration={start_iter})")
            except Exception as e:
                print(f"[RLTrainer] Warning: failed to resume from rl_last.pt: {e}")

        self.model.train()

        for iteration in range(start_iter, cfg.total_iterations + 1):
            self._reward_fn.notify_iteration(iteration)
            self.model.train()
            self._collect_rollout()

            kl_bc_coef = _compute_kl_bc_coef(iteration, cfg)
            avg_ppo_result = self._ppo_update(iteration=iteration, kl_bc_coef=kl_bc_coef)
            buffer_stats = self._buffer.episode_stats()

            self._rl_metrics_logger.log_train(iteration, avg_ppo_result, buffer_stats)
            self._buffer.clear()

            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

            if iteration % cfg.snapshot_every == 0:
                path = self._ckpt_manager.save_snapshot(
                    self.model, self.state_builder, self.codec, iteration
                )
                self._pool.add_snapshot(path, iteration)

            if iteration % cfg.save_every == 0:
                metrics = {
                    "policy_loss": avg_ppo_result.policy_loss,
                    "value_loss": avg_ppo_result.value_loss,
                    "entropy": avg_ppo_result.entropy,
                }
                self._ckpt_manager.save_rl_checkpoint(
                    self.model,
                    self._optimizer,
                    self._lr_scheduler,
                    self.state_builder,
                    self.codec,
                    iteration,
                    metrics,
                )

            if iteration % cfg.eval_every == 0:
                try:
                    from training.evaluation.evaluator import Evaluator
                    from bots.neural.bot import NeuralBot
                    bot = NeuralBot(
                        model=self.model,
                        state_builder=self.state_builder,
                        codec=self.codec,
                        device=cfg.device,
                    )
                    evaluator = Evaluator(
                        bot=bot,
                        opponents=cfg.eval_opponents,
                        n_matches=cfg.n_eval_matches,
                        run_dir=cfg.run_dir,
                    )
                    eval_results = evaluator.run(epoch=iteration)
                    self._rl_metrics_logger.log_eval(iteration, eval_results)
                    mean_winrate = sum(
                        v.get("win_rate", 0.0) for v in eval_results.values()
                    ) / max(len(eval_results), 1)
                    if mean_winrate > self._best_mean_winrate:
                        self._best_mean_winrate = mean_winrate
                        best_metrics = {
                            "policy_loss": avg_ppo_result.policy_loss,
                            "value_loss": avg_ppo_result.value_loss,
                            "entropy": avg_ppo_result.entropy,
                        }
                        self._ckpt_manager.save_rl_checkpoint(
                            self.model,
                            self._optimizer,
                            self._lr_scheduler,
                            self.state_builder,
                            self.codec,
                            iteration,
                            best_metrics,
                            is_best_winrate=True,
                        )
                except Exception as e:
                    print(f"[RLTrainer] Eval failed at iteration {iteration}: {e}")

            if iteration % 10 == 0:
                print(
                    f"[RLTrainer] iter={iteration}/{cfg.total_iterations} "
                    f"policy_loss={avg_ppo_result.policy_loss:.4f} "
                    f"value_loss={avg_ppo_result.value_loss:.4f} "
                    f"entropy={avg_ppo_result.entropy:.4f} "
                    f"episodes={buffer_stats['n_episodes']} "
                    f"mean_ep_reward={buffer_stats['mean_ep_reward']:.4f}"
                )
