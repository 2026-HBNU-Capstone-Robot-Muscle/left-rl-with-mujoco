"""MuJoCo finger robot environment definitions."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from reward_function import RewardConfig, RewardState, compute_reward


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "my-robot" / "scene.xml"
DEFAULT_LOG_DIR = ROOT / "runs" / "ppo"
FINGER_JOINT_NAMES = (
    "right-body-link1",
    "front-body-link1",
    "left-body-link1",
    "back-body-link1",
)


class FingerRobotEnv(gym.Env):
    """A task for closing the four tendon-driven fingers."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, render_mode: str | None = None):
        self.model_path = Path(model_path).resolve()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.max_steps = 5000
        self.step_count = 0
        self.viewer = None

        self.reward_cfg = RewardConfig()
        self.reward_state = RewardState()

        self.finger_joint_ids = np.array(
            [self.model.joint(name).id for name in FINGER_JOINT_NAMES],
            dtype=np.int32,
        )
        self.finger_qpos_addresses = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.finger_joint_ids],
            dtype=np.int32,
        )
        self.lower_limits = self.model.jnt_range[self.finger_joint_ids, 0]
        self.upper_limits = self.model.jnt_range[self.finger_joint_ids, 1]

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model.nq + self.model.nv,),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        return np.concatenate((self.data.qpos, self.data.qvel)).astype(np.float32)

    def _finger_progress(self) -> float:
        positions = self.data.qpos[self.finger_qpos_addresses]
        normalized = (positions - self.lower_limits) / (self.upper_limits - self.lower_limits)
        return float(np.mean(np.clip(normalized, 0.0, 1.0)))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # robot.xml's base body is welded to the world (no freejoint), so this
        # only applies if a freejoint actually exists at joint 0 - guard against
        # it instead of assuming nq >= 7 means "free base", which used to
        # silently overwrite the first 7 finger-joint angles.
        if self.model.njnt > 0 and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
            base_addr = self.model.jnt_qposadr[0]
            self.data.qpos[base_addr:base_addr + 7] = np.array(
                [0.0, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0]
            )

        self.data.qpos[self.finger_qpos_addresses] = self.lower_limits
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.reward_state = RewardState()
        return self._get_obs(), {"finger_progress": self._finger_progress()}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)

        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = clipped_action
        mujoco.mj_step(self.model, self.data, nstep=1)
        self.step_count += 1

        progress = self._finger_progress()

        reward, reward_breakdown, self.reward_state = compute_reward(
            self.model,
            self.data,
            clipped_action.astype(np.float64),
            self.reward_cfg,
            self.reward_state,
        )

        terminated = bool(progress > 0.98)
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        info = {"finger_progress": progress, "reward_breakdown": reward_breakdown}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def make_env(model_path: Path | str, render_mode: str | None = None):
    return Monitor(FingerRobotEnv(model_path=model_path, render_mode=render_mode))
