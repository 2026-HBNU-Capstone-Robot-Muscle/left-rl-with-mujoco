"""
4-finger tendon-coupled gripper - v5.0 (실기기(DYNAMIXEL) 스펙 대응판)
─────────────────────────────────────────────────────────────────
sim2real 이식 코드(run.py/core.py/apply_policy.py, stable-baselines3 PPO)를
그대로 분석해서 확인한 실기기 계약에 맞춰 v4에서 환경 인터페이스만 바꿈:

    obs(8)    = 텐던 길이 4개 + 텐던 각속도 4개  (라디안, 비정규화, 특권정보 없음)
    action(4) = [-1,1] 정규화 → 텐던 목표길이(position 액추에이터 ctrl)
    제어주기  = 100ms 명목 (real hardware 기준) = 물리 스텝 2ms x 50

v4(gripper_env_v4.py) 대비 바뀐 것:
  1. 액추에이터: torque motor -> position servo (실기기 DYNAMIXEL mode5,
     Current-based Position Control 모사). robot.xml에서 이미 교체함.
     ctrlrange=[0, 2.0943951023931953]. 2.0943951023931953 rad
     (=pi/2 + 3*10deg)는 완전 열림~완전 닫힘까지 텐던 길이 폭이며, 실측
     검증 결과 실기기 core.py의 CTRL 폭(right/front/left/back 전부
     2.0944)과 정확히 일치함 (finger별 low/high 오프셋 차이는 실기기의
     모터 영점 캘리브레이션이 흡수하므로 시뮬에서는 통일된 [0, span]
     하나만 쓰면 됨).
     kp=8, kv=0.3, forcerange=[-2,2]는 실기기의 전류 상한(Current-based
     Position Control)을 흉내낸 잠정값 — 실측 기반 재조정 필요(TODO).
  2. obs: v4의 61-dim(모터pos/vel 32 + 큐브상태 13 + tip 12 + touch 4)
     -> 8-dim(텐던 길이 4 + 텐던 각속도 4)로 축소. 실기기는 모터 인코더
     외에 큐브 포즈나 접촉힘을 볼 수 없으므로, 정책이 "보는" 정보를
     실기기와 동일하게 제한함.
  3. 보상 계산은 v4와 100% 동일 (reach/grasp/contact/rolloff 메인 +
     grip/contact/slip/energy/switch 보조). 보상은 학습에만 쓰이고
     정책에 노출되는 obs가 아니므로, 시뮬레이터 내부 특권 정보(큐브
     위치, touch 센서)를 그대로 계속 사용해도 됨 — obs를 축소한다고
     보상 설계까지 버릴 필요는 없음.
  4. n_frames=50: MJCF 물리 스텝 2ms x 50 = 100ms. 정책은 100ms마다
     한 번씩 호출되고, 그 사이 position 액추에이터가 목표를 계속
     추종하도록 물리 스텝만 50번 진행함. (v3 README에서 팀원이 지적한
     "로컬 시뮬 2ms vs 실기 100ms 불일치" 문제를 여기서 해소함.)

텐던 각속도(ten_velocity)는 MJX의 mjx.Data에 필드가 없어서, 고정 텐던의
정의(각 손가락 텐던 = coef 1.0인 관절 4개의 합, robot.xml에서 실측 확인)를
그대로 이용해 관절 각속도 4개를 더해서 계산함.
"""

import jax
from jax import numpy as jp
from mujoco import mjx
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State


NUM_FINGER_JOINTS = 16  # 4 fingers x 4 joints
NUM_ACTUATORS = 4
N_FINGER = 4
N_FRAMES = 15  # 2ms(물리 스텝) x 15 = 30ms(실기기 명목 제어주기)

TENDON_SPAN = 2.0943951023931953  # 완전열림(0)~완전닫힘(TENDON_SPAN), 실측 검증됨

# ── 도메인 랜덤화 ────────────────────────────────────────────────────
CUBE_XY_RANGE = 0.01
CUBE_YAW_RANGE = jp.pi / 4.0

# ── 메인 보상 (v2/v4와 동일) ─────────────────────────────────────────
REACH_DELTA_SCALE = 100.0
REACH_CLIP_LOW, REACH_CLIP_HIGH = -2.0, 5.0
REACH_SCALE = 5.0
GRASP_SLOW_SCALE = 5.0
GRASP_CLOSE_THRESHOLD = 0.06
GRASP_SCALE = 20.0
CONTACT_DEAD_ZONE = 0.005
CONTACT_FULL_ZONE = 0.03
CONTACT_DECAY = 0.015
CONTACT_SCALE = 5.0
ROLLOFF_FREE_RADIUS = 0.09
ROLLOFF_SCALE = 50.0
STEP_PENALTY = 0.1
DONE_PENALTY = 100.0
DONE_LATERAL = 0.14

# ── 보조 보상 (v4와 동일, 팀원 설계 가중치 하향) ─────────────────────
W_GRIP_AUX = 0.2
W_CONTACT_AUX = 0.2
W_SLIP_AUX = 0.1
W_ENERGY_AUX = 0.001
W_SWITCH_AUX = 0.02

TARGET_FORCE = 5.0
SIGMA_F = 3.0
SWITCH_THRESHOLD = 0.0
TOUCH_CONTACT_EPS = 1e-4


class TendonGripperEnvV5(PipelineEnv):

    def __init__(self, xml_path, **kwargs):
        model = mujoco.MjModel.from_xml_path(xml_path)
        model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        model.opt.iterations = 50
        model.opt.noslip_iterations = 25

        self.finger_qpos_idx = jp.arange(NUM_FINGER_JOINTS)
        self.finger_qvel_idx = jp.arange(NUM_FINGER_JOINTS)

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

        # 텐던 4개 = 각각 coef 1.0인 관절 4개의 합 (robot.xml 실측 확인,
        # right[0:4], front[4:8], left[8:12], back[12:16] 순서).
        # ten_velocity = qvel를 이 그룹별로 합산해서 계산.
        self._tendon_group_idx = jp.arange(NUM_FINGER_JOINTS).reshape(N_FINGER, 4)

        mjx_model = mjx.put_model(model)
        super().__init__(mjx_model, backend='mjx', n_frames=N_FRAMES, **kwargs)

    @property
    def action_size(self):
        return NUM_ACTUATORS

    @property
    def observation_size(self):
        return 8  # ten_length(4) + ten_velocity(4)

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

    def _get_ten_length_vel(self, pipeline_state):
        ten_length = pipeline_state.ten_length  # MJX가 직접 제공 (4,)
        qvel_fingers = pipeline_state.qvel[self.finger_qvel_idx].reshape(N_FINGER, 4)
        ten_velocity = jp.sum(qvel_fingers, axis=-1)  # coef=1.0 x 4관절 합
        return ten_length, ten_velocity

    def _get_obs(self, pipeline_state):
        """실기기 core.py의 observation()과 동일한 구성: 텐던 길이(rad) +
        텐던 각속도(rad/s), 정규화 없음. 순서는 right/front/left/back."""
        ten_length, ten_velocity = self._get_ten_length_vel(pipeline_state)
        return jp.concatenate([ten_length, ten_velocity])

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
                'reward_reach': 0.0,
                'reward_grasp': 0.0,
                'reward_contact': 0.0,
                'penalty_rolloff': 0.0,
                'min_tip_dist_per_step': jp.min(tip_dists0),
                'cube_height_per_step': cube_pos0[2],
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

        # action[-1,1] -> 텐던 목표길이[0, TENDON_SPAN] (core.py의 action_targets와
        # 동일한 선형매핑 규약. 실기기별 영점 오프셋은 실기기 쪽에서 흡수).
        a = jp.clip(action, -1.0, 1.0)
        ctrl = (a + 1.0) * 0.5 * TENDON_SPAN
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        obs = self._get_obs(pipeline_state)

        cube_pos, cube_quat, cube_linvel, cube_angvel = self._get_cube_state(pipeline_state)
        tip_pos = self._get_tip_positions(pipeline_state)
        tip_dists = jp.linalg.norm(tip_pos - cube_pos[None, :], axis=-1)
        min_tip_dist = jp.min(tip_dists)
        tip_dists_sum = jp.sum(tip_dists)

        cube_height = cube_pos[2]
        cube_lateral = jp.linalg.norm(cube_pos[:2] - self.hand_center[:2])

        # ── 메인 보상 (v2/v4와 동일) ─────────────────────────────────
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

        # ── 보조 보상 (v4와 동일, 특권 정보로 계산 — obs엔 노출 안 됨) ──
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
