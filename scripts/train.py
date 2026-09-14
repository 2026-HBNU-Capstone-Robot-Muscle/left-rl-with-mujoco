"""Training logic for PPO on the MuJoCo finger robot."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from finger_robot_env import make_env


class BestModelSaveCallback(BaseCallback):
    """Save the best-performing policy seen so far."""

    def __init__(self, log_dir: Path, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.best_model_path = self.log_dir / "finger_robot_ppo_best.zip"
        self.best_mean_reward = float("-inf")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is None:
                continue

            episode_reward = float(episode.get("r", float("-inf")))
            if episode_reward > self.best_mean_reward:
                self.best_mean_reward = episode_reward
                self.model.save(self.best_model_path)
                if self.verbose > 0:
                    print(
                        f"[BestModelSaveCallback] Saved best model with mean reward={episode_reward:.4f} "
                        f"at {self.best_model_path}"
                    )
        return True


def train(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.model, render_mode="human" if args.render_training else None)

    if args.check_env:
        check_env(env.unwrapped, warn=True)

    env = Monitor(env, filename=str(log_dir / "monitor"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    callback = BestModelSaveCallback(log_dir=log_dir, verbose=1)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(log_dir / "finger_robot_ppo_final")
    env.close()
