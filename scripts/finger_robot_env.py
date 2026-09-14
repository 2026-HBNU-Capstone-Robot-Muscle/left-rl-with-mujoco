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
# One fixed tendon per real motor. Order matches the actuator order in
# robot.xml/tendons.xml (and FINGER_MOTORS in reward_function.py), which is
# what matters for observations/actions to line up correctly.
FINGER_TENDON_NAMES = (
    "right_finger_tendon",
    "front_finger_tendon",
    "left_finger_tendon",
    "back_finger_tendon",
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

        # Real hardware only exposes 4 motor encoders (position + velocity),
        # not all 16 joint angles - the tendon length/velocity is exactly
        # what a real motor encoder reads (gear=1, so tendon length units
        # == motor rotation units). Use those as the observation instead of
        # the full joint-space qpos/qvel.
        self.tendon_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
                for name in FINGER_TENDON_NAMES
            ],
            dtype=np.int32,
        )

        # Actuators are now position servos (see robot.xml/tendons.xml): the
        # policy outputs a normalized action in [-1, 1] per motor, which is
        # rescaled in step() to that motor's physical ctrlrange (target
        # tendon length == target motor position). This mirrors a real motor
        # driver, which takes a position setpoint, not a torque/effort.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * len(FINGER_TENDON_NAMES),),  # 4 motor positions + 4 motor velocities
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        # Tendon length/velocity == what a real motor encoder would report
        # (position + velocity of each of the 4 motors), not the full
        # 16-joint qpos/qvel state.
        motor_position = self.data.ten_length[self.tendon_ids]
        motor_velocity = self.data.ten_velocity[self.tendon_ids]
        return np.concatenate((motor_position, motor_velocity)).astype(np.float32)

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

        # Rescale the normalized [-1, 1] action to each motor's physical
        # ctrlrange (target tendon length). Actuators are position servos
        # now, so ctrl is a target position, not a torque.
        ctrl_low = self.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.model.actuator_ctrlrange[:, 1]
        target_position = ctrl_low + (clipped_action + 1.0) * 0.5 * (ctrl_high - ctrl_low)

        self.data.ctrl[:] = target_position
        mujoco.mj_step(self.model, self.data, nstep=1)
        self.step_count += 1

        progress = self._finger_progress()

        reward, reward_breakdown, self.reward_state = compute_reward(
            self.model,
            self.data,
            clipped_action.astype(np.float64),  # normalized action; same [-1, 1] semantics reward_function.py expects
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
