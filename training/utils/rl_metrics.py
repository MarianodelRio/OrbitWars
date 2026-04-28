"""RLMetricsLogger: logs training and evaluation metrics for RL training."""

from __future__ import annotations

from pathlib import Path

from training.utils.metrics import MetricsLogger


class RLMetricsLogger:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        self._train_logger = MetricsLogger(
            path=metrics_dir / "rl_train.csv",
            fields=[
                "iteration",
                "total_loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "explained_variance",
                "mean_ep_reward",
                "n_episodes",
            ],
        )

        self._eval_logger = MetricsLogger(
            path=metrics_dir / "rl_eval.csv",
            fields=["iteration", "opponent", "win_rate", "draw_rate", "loss_rate"],
        )

    def log_train(self, iteration: int, ppo_result, buffer_stats: dict) -> None:
        row = {
            "iteration": iteration,
            "total_loss": getattr(ppo_result, "total_loss", ""),
            "policy_loss": getattr(ppo_result, "policy_loss", ""),
            "value_loss": getattr(ppo_result, "value_loss", ""),
            "entropy": getattr(ppo_result, "entropy", ""),
            "approx_kl": getattr(ppo_result, "approx_kl", ""),
            "clip_fraction": getattr(ppo_result, "clip_fraction", ""),
            "explained_variance": getattr(ppo_result, "explained_variance", ""),
            "mean_ep_reward": buffer_stats.get("mean_ep_reward", ""),
            "n_episodes": buffer_stats.get("n_episodes", ""),
        }
        self._train_logger.log(row)

    def log_eval(self, iteration: int, eval_results: dict) -> None:
        for opponent, result in eval_results.items():
            row = {
                "iteration": iteration,
                "opponent": opponent,
                "win_rate": result.get("win_rate", ""),
                "draw_rate": result.get("draw_rate", ""),
                "loss_rate": result.get("loss_rate", ""),
            }
            self._eval_logger.log(row)
