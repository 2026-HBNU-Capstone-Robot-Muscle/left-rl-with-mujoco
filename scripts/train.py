"""Training logic for PPO on the MuJoCo finger robot."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from finger_robot_env import make_env


def train(args: argparse.Namespace) -> None:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.model, render_mode="human" if args.render_training else None)

    if args.check_env:
        check_env(env.unwrapped, warn=True)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(log_dir / "finger_robot_ppo")
    env.close()
