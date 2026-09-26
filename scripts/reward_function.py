"""
Reward function implementation for the 4-finger MuJoCo grasping robot.

Implements:
    R_total = w_grip*R_grip + w_contact*R_contact + w_hold*R_full_hold
              - w_slip*P_slip - w_smooth*P_smooth - w_switch*P_switch

    R_grip      = 파지력 보상 (목표 파지력 F_target까지 선형 증가, max_safe_force까지 1.0 유지)
    R_contact   = N_contact / N_finger
    R_full_hold = 4개 손가락이 모두 최소 유지력(min_hold_force) 이상으로 접촉 시 보너스 (1.0)
    P_slip      = sum_i ||delta_p_i_relative|| (미끄러짐 페널티)
    P_smooth    = sum_i (a_i,t - a_i,t-1)^2 (행동 급변/Jerk 억제)
    P_switch    = sum_i [a_i,t and a_i,t-1 are on opposite sides of the open/close threshold]

Assumes a 4-finger tendon-driven gripper (right/front/left/back), matching
robot.xml + cube_fragment.xml from the current project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco


# ---------------------------------------------------------------------------
# Config - adjust to match your model's names
# ---------------------------------------------------------------------------

FINGER_MOTORS = [
    "right_finger_motor",
    "front_finger_motor",
    "left_finger_motor",
    "back_finger_motor",
]

CUBE_GEOM_NAMES = ["target_cube_body", "target_cube_top_layer"]

N_FINGER = len(FINGER_MOTORS)

# action sign threshold used to decide "closed" vs "open" for P_switch
SWITCH_THRESHOLD = 0.0


@dataclass
class RewardWeights:
    w_grip: float = 2.5       # R_grip: 목표 파지력 추종 보상
    w_contact: float = 1.0    # R_contact: 접촉 손가락 비율 보상
    w_hold: float = 3.0       # R_full_hold: 4지 동시 안정 파지 보너스
    w_slip: float = 0.5       # P_slip: 접촉점 미끄러짐 억제
    w_smooth: float = 0.1     # P_smooth: 행동 급변(Jerk) 억제
    w_switch: float = 0.5    # P_switch: 열림/닫힘 떨림 억제

    # 하위 호환용 속성
    @property
    def w1(self) -> float:
        return self.w_grip

    @property
    def w2(self) -> float:
        return self.w_contact

    @property
    def w3(self) -> float:
        return self.w_slip

    @property
    def w4(self) -> float:
        return self.w_smooth

    @property
    def w5(self) -> float:
        return self.w_switch


@dataclass
class RewardConfig:
    target_force: float = 6.0     # F_target: 큐브를 단단히 고정하기 위한 목표 파지력 (N)
    min_hold_force: float = 2.0   # 4지 동시 고정 판정을 위한 최소 개별 손가락 힘 (N)
    max_safe_force: float = 15.0  # 과도한 힘 제한 기준 (N)
    sigma_f: float = 2.0          # 초과 힘 감쇠 계수
    weights: RewardWeights = field(default_factory=RewardWeights)


@dataclass
class RewardState:
    """Carries state between steps (previous action, previous contact points)."""
    prev_actions: np.ndarray | None = None                 # shape (N_FINGER,)
    prev_contact_points: dict[str, np.ndarray] = field(default_factory=dict)  # finger -> world xyz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geom_id_set(model: mujoco.MjModel, names: list[str]) -> set[int]:
    ids = set()
    for name in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid != -1:
            ids.add(gid)
    return ids


# 모든 손가락 링크 바디(링크1~링크4)를 해당 모터로 매핑
BODY_TO_FINGER = {
    # Right finger links
    "part_1": "right_finger_motor",
    "part_1_2": "right_finger_motor",
    "part_1_3": "right_finger_motor",
    "part_1_4": "right_finger_motor",
    # Front finger links
    "part_1_5": "front_finger_motor",
    "part_1_6": "front_finger_motor",
    "part_1_7": "front_finger_motor",
    "part_1_8": "front_finger_motor",
    # Left finger links
    "part_1_9": "left_finger_motor",
    "part_1_10": "left_finger_motor",
    "part_1_11": "left_finger_motor",
    "part_1_12": "left_finger_motor",
    # Back finger links
    "part_1_13": "back_finger_motor",
    "part_1_14": "back_finger_motor",
    "part_1_15": "back_finger_motor",
    "part_1_16": "back_finger_motor",
}


def get_finger_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, cube_geom_ids: set[int]
) -> dict[str, dict]:
    """
    Scans mjData.contact for contacts between any robot finger geom and the cube.
    Returns per-finger contact information:
        contacts[finger] = {"pos": np.ndarray, "force": float}
    손가락의 여러 링크가 큐브에 닿는 경우 수직 항력을 합산합니다.
    """
    contacts: dict[str, dict] = {}

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2

        if g1 in cube_geom_ids or g2 in cube_geom_ids:
            other_geom = g2 if g1 in cube_geom_ids else g1
            body_id = model.geom_bodyid[other_geom]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            finger = BODY_TO_FINGER.get(body_name)
            if finger is None:
                continue

            pos = np.array(con.pos, dtype=np.float64).copy()

            # contact normal force magnitude (first element of 6D contact force)
            force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, force6)
            normal_force = abs(float(force6[0]))

            if finger in contacts:
                # 동일 손가락의 여러 링크가 접촉한 경우 힘 합산 및 위치 유지
                contacts[finger]["force"] += normal_force
            else:
                contacts[finger] = {"pos": pos, "force": normal_force}

    return contacts


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------

def compute_r_grip(contacts: dict[str, dict], cfg: RewardConfig) -> float:
    """
    R_grip: 파지력 보상 (0.0 ~ 1.0).
    - 접촉이 전혀 없으면 0.0 (허위 보상 방지).
    - 목표 힘 F_target(예: 6.0N)까지 힘이 강해질수록 보상이 선형 증가.
    - F_target ~ max_safe_force(예: 15.0N) 안정 파지 구간에서는 최대 보상 1.0 유지 (강한 파지 장려).
    - max_safe_force 초과 시에만 감쇠.
    """
    if not contacts:
        return 0.0

    f_t = float(np.mean([c["force"] for c in contacts.values()]))

    if f_t <= cfg.target_force:
        return float(np.clip(f_t / max(cfg.target_force, 1e-6), 0.0, 1.0))
    elif f_t <= cfg.max_safe_force:
        return 1.0
    else:
        return float(np.exp(-(f_t - cfg.max_safe_force) / cfg.sigma_f))


def compute_r_contact(contacts: dict[str, dict]) -> float:
    """R_contact = N_contact / N_finger."""
    return len(contacts) / N_FINGER


def compute_r_full_hold(contacts: dict[str, dict], cfg: RewardConfig) -> float:
    """
    R_full_hold: 4지 동시 고정 보너스.
    4개 손가락이 모두 큐브에 닿고, 각각 min_hold_force(1.5N) 이상으로 쥘 때 1.0 반환.
    """
    if len(contacts) < N_FINGER:
        return 0.0
    if all(c["force"] >= cfg.min_hold_force for c in contacts.values()):
        return 1.0
    return 0.0


def compute_p_slip(contacts: dict[str, dict], state: RewardState) -> float:
    """P_slip = sum_i ||delta_p_i_relative|| over fingers in contact both this step and last step."""
    total = 0.0
    for finger, c in contacts.items():
        prev = state.prev_contact_points.get(finger)
        if prev is not None:
            total += float(np.linalg.norm(c["pos"] - prev))
    return total


def compute_p_smooth(actions: np.ndarray, state: RewardState) -> float:
    """
    P_smooth = sum_i (a_i,t - a_i,t-1)^2.
    행동 크기 자체를 억제하여 손을 펴게 만들던 P_energy 대신,
    행동의 급격한 떨림/변화율(Jerk)을 억제하여 일정한 파지력을 지속 유지하도록 유도.
    """
    if state.prev_actions is None:
        return 0.0
    return float(np.sum((actions - state.prev_actions) ** 2))


def compute_p_switch(actions: np.ndarray, state: RewardState) -> float:
    """
    P_switch = number of fingers that crossed the open/close threshold since the previous step.
    """
    if state.prev_actions is None:
        return 0.0
    cur_closed = actions > SWITCH_THRESHOLD
    prev_closed = state.prev_actions > SWITCH_THRESHOLD
    return float(np.sum(cur_closed != prev_closed))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actions: np.ndarray,
    cfg: RewardConfig,
    state: RewardState,
) -> tuple[float, dict, RewardState]:
    """
    Call once per env.step(), after mj_step().

    Args:
        model, data: mujoco model/data.
        actions: array of shape (N_FINGER,), the ctrl just applied (order must match FINGER_MOTORS).
        cfg: weights + target force config.
        state: RewardState carried from the previous step.

    Returns:
        (total_reward, breakdown_dict, new_state)
    """
    cube_geom_ids = _geom_id_set(model, CUBE_GEOM_NAMES)
    contacts = get_finger_contacts(model, data, cube_geom_ids)

    r_grip = compute_r_grip(contacts, cfg)
    r_contact = compute_r_contact(contacts)
    r_full_hold = compute_r_full_hold(contacts, cfg)
    p_slip = compute_p_slip(contacts, state)
    p_smooth = compute_p_smooth(actions, state)
    p_switch = compute_p_switch(actions, state)

    w = cfg.weights
    total = (
        w.w_grip * r_grip
        + w.w_contact * r_contact
        + w.w_hold * r_full_hold
        - w.w_slip * p_slip
        - w.w_smooth * p_smooth
        - w.w_switch * p_switch
    )

    new_state = RewardState(
        prev_actions=actions.copy(),
        prev_contact_points={f: c["pos"] for f, c in contacts.items()},
    )

    mean_force = float(np.mean([c["force"] for c in contacts.values()])) if contacts else 0.0
    breakdown = {
        "R_grip": r_grip,
        "R_contact": r_contact,
        "R_full_hold": r_full_hold,
        "P_slip": p_slip,
        "P_energy": p_smooth,   # 하위 호환용 키 유지
        "P_smooth": p_smooth,
        "P_switch": p_switch,
        "R_total": total,
        "mean_force": mean_force,
        "n_contacts": len(contacts),
        "is_holding": bool(r_full_hold > 0.5),
    }
    return total, breakdown, new_state
