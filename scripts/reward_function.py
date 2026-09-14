"""
Reward function implementation for the 4-finger MuJoCo grasping robot.

Implements:
    R_total = w1*R_grip + w2*R_contact - w3*P_slip - w4*P_energy - w5*P_switch

    R_grip    = exp(-|F_t - F_target| / sigma_F)
    R_contact = N_contact / N_finger
    P_slip    = sum_i ||delta_p_i_relative||
    P_energy  = sum_i |a_i,t|
    P_switch  = sum_i [a_i,t and a_i,t-1 are on opposite sides of the open/close threshold]

Assumes a 4-finger tendon-driven gripper (right/front/left/back), matching
robot.xml + cube_fragment.xml from the current project:
    - actuators named "<finger>_finger_motor", ctrl range [-1, 1]
    - a cube body called "target_cube" with geoms "target_cube_body" /
      "target_cube_top_layer"

This module is intentionally self-contained (only needs `model` and `data`
from mujoco) so it can be dropped into any Gymnasium-style env's step()
function. Adjust FINGER_MOTORS / CUBE_GEOMS below if your naming differs.
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
    w1: float = 1.0   # R_grip
    w2: float = 1.0   # R_contact
    w3: float = 0.5   # P_slip
    w4: float = 0.005  # P_energy
    w5: float = 0.1   # P_switch


@dataclass
class RewardConfig:
    target_force: float = 1.0   # F_target
    sigma_f: float = 1.0        # sigma_F, normalizes force error
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


def _finger_contact_geom_prefix(finger: str) -> str:
    """Best-effort mapping from finger name -> the geoms belonging to that
    finger's last link (fingertip). Adjust if your geom naming differs.
    e.g. 'right_finger_motor' -> geoms under body 'part_1_4'."""
    return finger


def get_finger_contacts(
    model: mujoco.MjModel, data: mujoco.MjData, cube_geom_ids: set[int]
) -> dict[str, dict]:
    """
    Scans mjData.contact for contacts between any robot geom and the cube.
    Returns, per finger index (0..3), whether it's in contact and the world
    contact position (used for slip calculation).

    NOTE: this groups contacts by *body id* nearest each cube contact, not by
    a strict finger->geom map, since fingertip geom names weren't provided.
    Replace `body_to_finger` below with the exact fingertip body names from
    your model (e.g. the last <body> in each kinematic chain: part_1_4,
    part_1_8, part_1_12, part_1_16) for exact per-finger attribution.
    """
    body_to_finger = {
        "part_1_4": "right_finger_motor",
        "part_1_8": "front_finger_motor",
        "part_1_12": "left_finger_motor",
        "part_1_16": "back_finger_motor",
    }

    contacts: dict[str, dict] = {}

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2

        if g1 in cube_geom_ids or g2 in cube_geom_ids:
            other_geom = g2 if g1 in cube_geom_ids else g1
            body_id = model.geom_bodyid[other_geom]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

            finger = body_to_finger.get(body_name)
            if finger is None:
                continue

            pos = np.array(con.pos, dtype=np.float64).copy()

            # contact normal force magnitude (first element of 6D contact force)
            force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, force6)
            normal_force = abs(force6[0])

            contacts[finger] = {"pos": pos, "force": normal_force}

    return contacts


# ---------------------------------------------------------------------------
# Reward terms
# ---------------------------------------------------------------------------

def compute_r_grip(contacts: dict[str, dict], cfg: RewardConfig) -> float:
    """R_grip = exp(-|F_t - F_target| / sigma_F), F_t = mean contact force."""
    if not contacts:
        f_t = 0.0
    else:
        f_t = float(np.mean([c["force"] for c in contacts.values()]))
    return float(np.exp(-abs(f_t - cfg.target_force) / cfg.sigma_f))


def compute_r_contact(contacts: dict[str, dict]) -> float:
    """R_contact = N_contact / N_finger."""
    return len(contacts) / N_FINGER


def compute_p_slip(contacts: dict[str, dict], state: RewardState) -> float:
    """P_slip = sum_i ||delta_p_i_relative|| over fingers in contact both
    this step and last step."""
    total = 0.0
    for finger, c in contacts.items():
        prev = state.prev_contact_points.get(finger)
        if prev is not None:
            total += float(np.linalg.norm(c["pos"] - prev))
    return total


def compute_p_energy(actions: np.ndarray) -> float:
    """P_energy = sum_i |a_i,t|."""
    return float(np.sum(np.abs(actions)))


def compute_p_switch(actions: np.ndarray, state: RewardState) -> float:
    """P_switch = number of fingers that crossed the open/close threshold
    since the previous step."""
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
        actions: array of shape (N_FINGER,), the ctrl just applied
                 (order must match FINGER_MOTORS).
        cfg: weights + target force config.
        state: RewardState carried from the previous step (mutated/returned).

    Returns:
        (total_reward, breakdown_dict, new_state)
    """
    cube_geom_ids = _geom_id_set(model, CUBE_GEOM_NAMES)
    contacts = get_finger_contacts(model, data, cube_geom_ids)

    r_grip = compute_r_grip(contacts, cfg)
    r_contact = compute_r_contact(contacts)
    p_slip = compute_p_slip(contacts, state)
    p_energy = compute_p_energy(actions)
    p_switch = compute_p_switch(actions, state)

    w = cfg.weights
    total = (
        w.w1 * r_grip
        + w.w2 * r_contact
        - w.w3 * p_slip
        - w.w4 * p_energy
        - w.w5 * p_switch
    )

    new_state = RewardState(
        prev_actions=actions.copy(),
        prev_contact_points={f: c["pos"] for f, c in contacts.items()},
    )

    breakdown = {
        "R_grip": r_grip,
        "R_contact": r_contact,
        "P_slip": p_slip,
        "P_energy": p_energy,
        "P_switch": p_switch,
        "R_total": total,
    }
    return total, breakdown, new_state
