"""Load and visually test the MuJoCo scene.xml file."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "my-robot" / "scene.xml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="Path to the MuJoCo XML scene file to test.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional number of simulation steps to run before exiting. If omitted, runs until Ctrl+C is pressed.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Seconds to wait between steps for the viewer to refresh.",
    )
    return parser


def test_scene(scene_path: Path, steps: int | None = None, sleep_seconds: float = 0.01) -> None:
    scene_path = scene_path.resolve()
    print(f"Loading scene: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print("Launching MuJoCo viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)

    try:
        if steps is None:
            print("Running until Ctrl+C is pressed...")
            step_count = 0
            while True:
                mujoco.mj_step(model, data, nstep=1)
                viewer.sync()
                time.sleep(sleep_seconds)
                step_count += 1
        else:
            print(f"Running for {steps} steps...")
            for _ in range(steps):
                mujoco.mj_step(model, data, nstep=1)
                viewer.sync()
                time.sleep(sleep_seconds)

            print(f"Scene test completed successfully for {steps} steps.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Closing viewer...")
    finally:
        viewer.close()
        print("Viewer closed.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    test_scene(args.scene, steps=args.steps, sleep_seconds=args.sleep)


if __name__ == "__main__":
    main()
