"""CLI entrypoint for PPO training on the MuJoCo finger robot."""

from __future__ import annotations

import argparse
from pathlib import Path

from finger_robot_env import DEFAULT_LOG_DIR, DEFAULT_MODEL
from train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--timesteps", type=int, default=100_000)   # 테스트를 위해 20회를 기본으로 설정
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--render-training", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


# GUI를 사용하여 학습 과정 확인하는 방법
# python scripts/main.py --render-training --timesteps 100000
if __name__ == "__main__":
    main()
