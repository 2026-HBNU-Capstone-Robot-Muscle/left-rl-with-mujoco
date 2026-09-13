"""Evaluate a trained PPO policy on the MuJoCo finger robot."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from finger_robot_env import DEFAULT_LOG_DIR, DEFAULT_MODEL, make_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_LOG_DIR / "finger_robot_ppo")
    parser.add_argument("--eval-steps", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    return parser


def evaluate(args: argparse.Namespace) -> None:
    env = make_env(args.model, render_mode="human")
    model = PPO.load(args.model_file, env=env, device=args.device)

    observation, _ = env.reset()
    for _ in range(args.eval_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            observation, _ = env.reset()

    env.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
