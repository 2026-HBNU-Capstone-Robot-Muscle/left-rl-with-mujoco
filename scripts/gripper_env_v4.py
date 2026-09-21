"""
4-finger tendon-coupled gripper - v4.0 (메인+보조 보상 하이브리드)
─────────────────────────────────────────────────────────────────
25M 스텝 학습(v3.0)이 "접촉 0회, 가만히 있기"로 실패한 것을 확인한 뒤
(training_log_v3.txt 분석: R_grip/R_contact/touch가 전 구간 정확히 0.00),
원인을 다음과 같이 진단함:

    팀원 설계(R_grip + R_contact - P_slip - P_energy - P_switch)는
    "이미 접촉된 상태를 다듬는" 보상이지, "접촉 전 접근을 유도"하는
    dense reward가 전혀 없음. 무작위 텐던 토크 탐색만으로는 우연히도
    단 한번도 접촉이 안 일어났고, 에이전트는 P_energy/P_switch만
    줄이는 로컬 옵티멈(정지)에 빠졌음.

v4.0의 해결 방향 (사용자 확정):
  1. 메인 보상은 예전 gripper_env_v2.py 스타일로 되돌림
     - reward_reach   : 손끝-큐브 거리가 줄어드는 방향으로 dense guidance
     - reward_grasp   : 손끝이 큐브에 가깝고 느리게(정지-접촉) 움직일 때
     - reward_contact : 큐브가 "적당히" 변위(눌림/쥐어짐)했을 때
     - penalty_rolloff: 큐브가 손 중심에서 너무 멀리 밀려나면 페널티
     즉 접근 유도(dense) 신호를 다시 살려서 탐색이 접촉까지 도달하게 함.

  2. 팀원 설계(R_grip/R_contact/P_slip/P_energy/P_switch)는 보조 보상으로
     격하. 가중치를 v3.0 대비 대폭 낮춰서(약 1/5), "메인 보상이 만든
     접촉 상황을 힘 조절/에너지 효율 방향으로 다듬는" 역할만 하도록 함.
     절대 메인 보상의 부호를 뒤집을 만큼 크지 않게 설계.

  3. 큐브 초기 위치(x,y)와 초기 회전(yaw)을 도메인 랜덤화로 유지.
     - xy: 기존과 동일 (±0.01m)
     - yaw: 기존 ±0.35rad(≈20°) -> ±45°(π/4 rad)로 확장.
       정육면체는 90° 회전마다 자기 자신과 겹치는 대칭성이 있으므로,
       ±45°가 "모든 서로 다른 회전 상태"를 커버하는 자연스러운 범위.

  4. [버그 수정] v2의 reward_grasp `is_close` 임계값(0.025m)은 과거
     다른(더 작은) 큐브 기준으로 잡힌 값으로 보임. 지금 쓰는 56mm
     큐브(반폭 0.028m)는 손끝이 표면에 완전히 닿아도 중심까지 거리가
     최소 0.028m이기 때문에, 임계값 0.025m로는 접촉 상태에서도
     is_close=0이 되어 grasp 보상이 사실상 죽어있었음. 큐브 반폭 +
     여유(코너 대각선 0.0485m 고려)를 반영해 0.06m로 재조정함.

obs/센서 구조, touch_zone 등 물리·인터페이스 쪽은 v3.0(gripper_env_v3.py)
구조를 그대로 사용 (touch 센서, tip site, x/y-slide+z-hinge 큐브).
"""

import jax
from jax import numpy as jp
from mujoco import mjx
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State


NUM_FINGER_JOINTS = 16  # 4 fingers x 4 joints
NUM_ACTUATORS = 4       # 1 tendon motor per finger
N_FINGER = 4

# ── 도메인 랜덤화 범위 ───────────────────────────────────────────────
CUBE_XY_RANGE = 0.01          # 기존과 동일 (±1cm)
CUBE_YAW_RANGE = jp.pi / 4.0  # ±45deg. 정육면체 90도 대칭을 고려해 확장.

# ── 메인 보상 (gripper_env_v2.py 그대로 이식) ───────────────────────
# reward_reach
REACH_DELTA_SCALE = 100.0
REACH_CLIP_LOW, REACH_CLIP_HIGH = -2.0, 5.0
REACH_SCALE = 5.0
# reward_grasp
GRASP_SLOW_SCALE = 5.0
GRASP_CLOSE_THRESHOLD = 0.06   # [버그 수정] 0.025 -> 0.06 (56mm 큐브 반폭 0.028m + 여유 반영)
GRASP_SCALE = 20.0
# reward_contact (큐브 변위 기반)
CONTACT_DEAD_ZONE = 0.005
CONTACT_FULL_ZONE = 0.03
CONTACT_DECAY = 0.015
CONTACT_SCALE = 5.0
# penalty_rolloff
ROLLOFF_FREE_RADIUS = 0.09
ROLLOFF_SCALE = 50.0
STEP_PENALTY = 0.1
DONE_PENALTY = 100.0
DONE_LATERAL = 0.14

# ── 보조 보상 (gripper_env_v3.py 팀원 설계, 가중치 대폭 하향) ───────
# v3.0 원래 가중치: W_GRIP=1.0, W_CONTACT=1.0, W_SLIP=0.5, W_ENERGY=0.005, W_SWITCH=0.1
# 메인 보상(reach~25, grasp~20, contact~5 스케일)을 절대 압도하지 않도록
# 약 1/5 수준으로 낮춤. "이미 만들어진 접촉을 다듬는" 역할만 하게 함.
W_GRIP_AUX = 0.2
W_CONTACT_AUX = 0.2
W_SLIP_AUX = 0.1
W_ENERGY_AUX = 0.001
W_SWITCH_AUX = 0.02

TARGET_FORCE = 5.0
SIGMA_F = 3.0
SWITCH_THRESHOLD = 0.0
TOUCH_CONTACT_EPS = 1e-4


class TendonGripperEnv(PipelineEnv):

    def __init__(self, xml_path, **kwargs):
        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        model.opt.iterations = 50
        model.opt.noslip_iterations = 25

        self.finger_qpos_idx = jp.arange(NUM_FINGER_JOINTS)
        self.finger_qvel_idx = jp.arange(NUM_FINGER_JOINTS)

        jnt_range = np.array(model.jnt_range[:NUM_FINGER_JOINTS])
        self.jnt_low = jp.array(jnt_range[:, 0])
        self.jnt_high = jp.array(jnt_range[:, 1])

        slide_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_slide_x')
        self.cube_qpos_start = int(model.jnt_qposadr[slide_x_id])
        self.cube_qvel_start = int(model.jnt_dofadr[slide_x_id])

        self.tip_site_ids = jp.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip'),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip_2'),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip_3'),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip_4'),
        ])

        touch_names = ['touch_tip', 'touch_tip_2', 'touch_tip_3', 'touch_tip_4']
        touch_adr = []
        for name in touch_names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            assert sid != -1, f"센서 '{name}'를 robot_box_scene.xml에서 못 찾음"
            touch_adr.append(int(model.sensor_adr[sid]))
        self.touch_sensor_idx = jp.array(touch_adr)

        self.cube_fixed_z = float(model.body_pos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        ][2])

        self.hand_center = jp.array([0.0, 0.0, 0.0])

        mjx_model = mjx.put_model(model)
        super().__init__(mjx_model, backend='mjx', **kwargs)

    @property
    def action_size(self):
        return NUM_ACTUATORS

    @property
    def observation_size(self):
        # motor_pos(16) + motor_vel(16) + cube_pos(3) + cube_quat(4)
        # + cube_linvel(3) + cube_angvel(3) + tip_pos(4x3=12) + touch(4)
        return 16 + 16 + 3 + 4 + 3 + 3 + 12 + 4

    def _get_tip_positions(self, pipeline_state):
        return pipeline_state.site_xpos[self.tip_site_ids]

    def _get_touch(self, pipeline_state):
        return pipeline_state.sensordata[self.touch_sensor_idx]

    def _get_cube_state(self, pipeline_state):
        xy = pipeline_state.qpos[self.cube_qpos_start:self.cube_qpos_start + 2]
        yaw = pipeline_state.qpos[self.cube_qpos_start + 2]
        cp = jp.concatenate([xy, jp.array([self.cube_fixed_z])])
        cq = jp.array([jp.cos(yaw / 2.0), 0.0, 0.0, jp.sin(yaw / 2.0)])
        xy_vel = pipeline_state.qvel[self.cube_qvel_start:self.cube_qvel_start + 2]
        yaw_vel = pipeline_state.qvel[self.cube_qvel_start + 2]
        clv = jp.concatenate([xy_vel, jp.array([0.0])])
        cav = jp.array([0.0, 0.0, yaw_vel])
        return cp, cq, clv, cav

    def _get_obs(self, pipeline_state):
        motor_pos = pipeline_state.qpos[self.finger_qpos_idx]
        motor_vel = pipeline_state.qvel[self.finger_qvel_idx]
        cube_pos, cube_quat, cube_linvel, cube_angvel = self._get_cube_state(pipeline_state)
        tip_pos = self._get_tip_positions(pipeline_state).reshape(-1)
        touch = self._get_touch(pipeline_state)

        motor_pos_norm = 2.0 * (motor_pos - self.jnt_low) / (self.jnt_high - self.jnt_low + 1e-8) - 1.0

        return jp.concatenate([
            motor_pos_norm, motor_vel,
            cube_pos, cube_quat, cube_linvel, cube_angvel,
            tip_pos, touch,
        ])

    def reset(self, rng):
        rng, key_x, key_y, key_yaw = jax.random.split(rng, 4)

        qpos = jp.array(self.sys.qpos0)
        qpos = qpos.at[self.cube_qpos_start].add(
            jax.random.uniform(key_x, (), minval=-CUBE_XY_RANGE, maxval=CUBE_XY_RANGE))
        qpos = qpos.at[self.cube_qpos_start + 1].add(
            jax.random.uniform(key_y, (), minval=-CUBE_XY_RANGE, maxval=CUBE_XY_RANGE))
        qpos = qpos.at[self.cube_qpos_start + 2].add(
            jax.random.uniform(key_yaw, (), minval=-CUBE_YAW_RANGE, maxval=CUBE_YAW_RANGE))

        pipeline_state = self.pipeline_init(qpos, jp.zeros(self.sys.nv))

        cube_pos0, _, _, _ = self._get_cube_state(pipeline_state)
        tip_pos0 = self._get_tip_positions(pipeline_state)
        tip_dists0 = jp.linalg.norm(tip_pos0 - cube_pos0[None, :], axis=-1)
        zero_action = jp.zeros(NUM_ACTUATORS)

        return State(
            pipeline_state,
            self._get_obs(pipeline_state),
            reward=0.0,
            done=0.0,
            metrics={
                'reward': 0.0,
                # 메인 보상
                'reward_reach': 0.0,
                'reward_grasp': 0.0,
                'reward_contact': 0.0,
                'penalty_rolloff': 0.0,
                'min_tip_dist_per_step': jp.min(tip_dists0),
                'cube_height_per_step': cube_pos0[2],
                # 보조 보상 (팀원 설계, 가중치 하향)
                'R_grip_aux': 0.0,
                'R_contact_aux': 0.0,
                'P_slip_aux': 0.0,
                'P_energy_aux': 0.0,
                'P_switch_aux': 0.0,
                'mean_touch_force_per_step': 0.0,
            },
            info={
                'rng': rng,
                'initial_cube_pos': cube_pos0,
                'prev_tip_dists_sum': jp.sum(tip_dists0),
                'prev_action': zero_action,
                'prev_tip_pos': tip_pos0,
                'prev_contact_mask': jp.zeros(N_FINGER, dtype=bool),
            },
        )

    def step(self, state, action):
        rng, _ = jax.random.split(state.info['rng'])
        initial_cube_pos = state.info['initial_cube_pos']
        prev_tip_sum = state.info['prev_tip_dists_sum']
        prev_action = state.info['prev_action']
        prev_tip_pos = state.info['prev_tip_pos']
        prev_contact_mask = state.info['prev_contact_mask']

        ctrl = jp.clip(action, -1.0, 1.0)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        obs = self._get_obs(pipeline_state)

        cube_pos, cube_quat, cube_linvel, cube_angvel = self._get_cube_state(pipeline_state)
        tip_pos = self._get_tip_positions(pipeline_state)
        tip_dists = jp.linalg.norm(tip_pos - cube_pos[None, :], axis=-1)
        min_tip_dist = jp.min(tip_dists)
        tip_dists_sum = jp.sum(tip_dists)

        cube_height = cube_pos[2]
        cube_lateral = jp.linalg.norm(cube_pos[:2] - self.hand_center[:2])

        # ══════════════════════════════════════════════════════════════
        # 메인 보상 (v2 스타일: 접근 유도 + 그립 + 변위 기반 접촉 + 이탈 방지)
        # ══════════════════════════════════════════════════════════════
        proximity_delta = (prev_tip_sum - tip_dists_sum) * REACH_DELTA_SCALE
        reward_reach = jp.clip(proximity_delta, REACH_CLIP_LOW, REACH_CLIP_HIGH) * REACH_SCALE

        motor_vel = pipeline_state.qvel[self.finger_qvel_idx]
        is_slow = jp.exp(-GRASP_SLOW_SCALE * jp.mean(jp.abs(motor_vel)))
        is_close = jp.clip((GRASP_CLOSE_THRESHOLD - min_tip_dist) / GRASP_CLOSE_THRESHOLD, 0.0, 1.0)
        reward_grasp = is_close * is_slow * GRASP_SCALE

        cube_displacement = jp.linalg.norm(cube_pos - initial_cube_pos)
        contact_signal = jp.where(
            cube_displacement < CONTACT_DEAD_ZONE, 0.0,
            jp.where(cube_displacement < CONTACT_FULL_ZONE, 1.0,
                     jp.exp(-(cube_displacement - CONTACT_FULL_ZONE) / CONTACT_DECAY))
        )
        reward_contact = contact_signal * CONTACT_SCALE

        penalty_rolloff = jp.maximum(cube_lateral - ROLLOFF_FREE_RADIUS, 0.0) * ROLLOFF_SCALE

        main_reward = (
            reward_reach + reward_grasp + reward_contact
            - penalty_rolloff - STEP_PENALTY
        )

        too_far = cube_lateral > DONE_LATERAL
        done = jp.where(too_far, 1.0, 0.0)
        main_reward -= jp.where(done > 0, DONE_PENALTY, 0.0)

        # ══════════════════════════════════════════════════════════════
        # 보조 보상 (팀원 설계: 힘 조절 / 미끄러짐 / 에너지 효율 / 스위칭)
        # 가중치를 낮춰서 메인 보상이 만든 접촉을 "다듬는" 정도로만 기여.
        # ══════════════════════════════════════════════════════════════
        touch = self._get_touch(pipeline_state)
        contact_mask = touch > TOUCH_CONTACT_EPS
        n_contact = jp.sum(contact_mask)
        has_contact = n_contact > 0

        f_t = jp.sum(touch * contact_mask) / jp.maximum(n_contact, 1)
        r_grip_aux = jp.where(
            has_contact,
            jp.exp(-jp.abs(f_t - TARGET_FORCE) / SIGMA_F),
            0.0,
        )
        r_contact_aux = n_contact / N_FINGER

        tip_delta = jp.linalg.norm(tip_pos - prev_tip_pos, axis=-1)
        slip_mask = contact_mask & prev_contact_mask
        p_slip_aux = jp.sum(tip_delta * slip_mask)

        p_energy_aux = jp.sum(jp.abs(action))
        cur_closed = action > SWITCH_THRESHOLD
        prev_closed = prev_action > SWITCH_THRESHOLD
        p_switch_aux = jp.sum((cur_closed != prev_closed).astype(jp.float32))

        aux_reward = (
            W_GRIP_AUX * r_grip_aux
            + W_CONTACT_AUX * r_contact_aux
            - W_SLIP_AUX * p_slip_aux
            - W_ENERGY_AUX * p_energy_aux
            - W_SWITCH_AUX * p_switch_aux
        )

        total_reward = main_reward + aux_reward

        new_metrics = {
            'reward': total_reward,
            'reward_reach': reward_reach,
            'reward_grasp': reward_grasp,
            'reward_contact': reward_contact,
            'penalty_rolloff': penalty_rolloff,
            'min_tip_dist_per_step': min_tip_dist,
            'cube_height_per_step': cube_height,
            'R_grip_aux': r_grip_aux,
            'R_contact_aux': r_contact_aux,
            'P_slip_aux': p_slip_aux,
            'P_energy_aux': p_energy_aux,
            'P_switch_aux': p_switch_aux,
            'mean_touch_force_per_step': jp.mean(touch),
        }

        new_info = state.info.copy()
        new_info['rng'] = rng
        new_info['prev_tip_dists_sum'] = tip_dists_sum
        new_info['prev_action'] = action
        new_info['prev_tip_pos'] = tip_pos
        new_info['prev_contact_mask'] = contact_mask

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=total_reward,
            done=done,
            metrics=new_metrics,
            info=new_info,
        )
