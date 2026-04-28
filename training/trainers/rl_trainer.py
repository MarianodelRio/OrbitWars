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
            clip_abs=cfg.reward_clip_abs,
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

        torch.manual_seed(cfg.seed)

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

        while not buffer.is_full():
            pf = torch.tensor(state["planet_features"], dtype=torch.float32).unsqueeze(0).to(device)
            ff = torch.tensor(state["fleet_features"], dtype=torch.float32).unsqueeze(0).to(device)
            fm = torch.tensor(state["fleet_mask"], dtype=torch.bool).unsqueeze(0).to(device)
            gf = torch.tensor(state["global_features"], dtype=torch.float32).unsqueeze(0).to(device)
            pm = torch.tensor(state["planet_mask"], dtype=torch.bool).unsqueeze(0).to(device)

            rl_masks = sampler.build_masks(state["context"], device=device)

            with torch.no_grad():
                output = self.model(pf, ff, fm, gf, pm)
                output_squeezed = PlanetPolicyOutput(
                    action_type_logits=output.action_type_logits.squeeze(0),
                    target_logits=output.target_logits.squeeze(0),
                    amount_logits=output.amount_logits.squeeze(0),
                    value=output.value,
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
            )
            buffer.add(roll_step)

            if done:
                opponent_fn = self._pool.sample(
                    self_play_prob=self.config.self_play_prob,
                    current_model_fn=self._current_model_as_agent(),
                )
                env.set_opponent(opponent_fn)
                state, _ = env.reset()
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
                output = self.model(pf, ff, fm, gf, pm)
                last_value = output.value.squeeze().item()

        self._buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

    def _ppo_update(self) -> PPOLossResult:
        cfg = self.config
        all_results = []
        for _epoch in range(cfg.ppo_epochs):
            batches = self._buffer.get_batches(cfg.ppo_batch_size, cfg.device)
            for batch in batches:
                self._optimizer.zero_grad()
                loss, result = compute_ppo_loss(self.model, batch, cfg)
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
            self.model.train()
            self._collect_rollout()

            avg_ppo_result = self._ppo_update()
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
