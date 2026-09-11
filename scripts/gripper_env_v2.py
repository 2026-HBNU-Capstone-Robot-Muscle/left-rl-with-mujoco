"""
4-finger tendon-coupled gripper grasping environment - v2.1
─────────────────────────────────────────────────────────────────
v2.0 대비 변경점:
  - robot.xml에 손가락 자기 자신 링크끼리의 self-collision exclude 추가
    (없어서 굽힘 자체가 물리적으로 막혀있던 버그 수정)
  - 큐브를 6-DOF freejoint(자유낙하) -> x,y 슬라이드 2-DOF로 변경
    (높이/회전 고정, 바닥에 놓인 것처럼 위치만 변함, 중력에 안 떨어짐)
  - 큐브 높이 z=-0.08m: 4손가락을 완전히 오므렸을 때 손끝이 모이는
    실측 위치로 고정
"""

import jax
from jax import numpy as jp
from mujoco import mjx
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State


NUM_FINGER_JOINTS = 16  # 4 fingers x 4 joints
NUM_ACTUATORS = 4        # 1 tendon motor per finger
CUBE_FIXED_Z = -0.08      # 큐브 높이는 항상 고정
CUBE_FIXED_QUAT = jp.array([1.0, 0.0, 0.0, 0.0])  # 회전도 항상 고정(identity)


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

        # 큐브는 x,y 슬라이드 조인트 2개 (freejoint 아님)
        slide_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cube_slide_x')
        self.cube_qpos_start = int(model.jnt_qposadr[slide_x_id])  # x,y가 연속 인덱스라고 가정
        self.cube_qvel_start = int(model.jnt_dofadr[slide_x_id])

        self.tip_site_ids = jp.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip'),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip_2'),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip_3'),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tip_4'),
        ])

        self.hand_center = jp.array([0.0, 0.0, 0.0])

        mjx_model = mjx.put_model(model)
        super().__init__(mjx_model, backend='mjx', **kwargs)

    @property
    def action_size(self):
        return NUM_ACTUATORS

    @property
    def observation_size(self):
        # motor_pos(16) + motor_vel(16) + cube_pos(3) + cube_quat(4)
        # + cube_linvel(3) + cube_angvel(3) + tip_pos(4*3=12)
        # (cube_pos의 z, cube_quat, angvel, linvel의 z는 항상 상수지만
        #  나중에 freejoint로 되돌릴 걸 감안해 인터페이스는 유지)
        return 16 + 16 + 3 + 4 + 3 + 3 + 12

    def _sample_domain_params(self, rng):
        keys = jax.random.split(rng, 2)
        cube_fric = jax.random.uniform(keys[0], (), minval=1.0, maxval=2.5)
        cube_mass_scale = jax.random.uniform(keys[1], (), minval=0.7, maxval=1.4)
        return {'cube_fric': cube_fric, 'cube_mass_scale': cube_mass_scale}

    def _get_tip_positions(self, pipeline_state):
        return pipeline_state.site_xpos[self.tip_site_ids]

    def _get_cube_state(self, pipeline_state):
        # x, y만 실제 DOF. z는 고정값, 회전/각속도도 항상 0.
        xy = pipeline_state.qpos[self.cube_qpos_start:self.cube_qpos_start + 2]
        cp = jp.concatenate([xy, jp.array([CUBE_FIXED_Z])])
        cq = CUBE_FIXED_QUAT
        xy_vel = pipeline_state.qvel[self.cube_qvel_start:self.cube_qvel_start + 2]
        clv = jp.concatenate([xy_vel, jp.array([0.0])])
        cav = jp.zeros(3)
        return cp, cq, clv, cav

    def _get_obs(self, pipeline_state):
        motor_pos = pipeline_state.qpos[self.finger_qpos_idx]
        motor_vel = pipeline_state.qvel[self.finger_qvel_idx]
        cube_pos, cube_quat, cube_linvel, cube_angvel = self._get_cube_state(pipeline_state)
        tip_pos = self._get_tip_positions(pipeline_state).reshape(-1)

        # 관절각 정규화: [low,high] -> [-1,1]
        motor_pos_norm = 2.0 * (motor_pos - self.jnt_low) / (self.jnt_high - self.jnt_low + 1e-8) - 1.0

        return jp.concatenate([
            motor_pos_norm, motor_vel,
            cube_pos, cube_quat, cube_linvel, cube_angvel,
            tip_pos,
        ])

    def reset(self, rng):
        rng, key_pos, key_dr = jax.random.split(rng, 3)
        dr = self._sample_domain_params(key_dr)

        new_model = self.sys.tree_replace({
            'geom_friction': self.sys.geom_friction.at[:].set(dr['cube_fric'])
        })

        qpos = jp.array(self.sys.qpos0)

        # 큐브 위치 살짝 랜덤화 (xy 평면)
        rng_x, rng_y = jax.random.split(key_pos)
        qpos = qpos.at[self.cube_qpos_start].add(
            jax.random.uniform(rng_x, (), minval=-0.01, maxval=0.01))
        qpos = qpos.at[self.cube_qpos_start + 1].add(
            jax.random.uniform(rng_y, (), minval=-0.01, maxval=0.01))

        pipeline_state = self.pipeline_init(qpos, jp.zeros(self.sys.nv))

        cube_pos0, _, _, _ = self._get_cube_state(pipeline_state)
        tip_pos0 = self._get_tip_positions(pipeline_state)
        tip_dists0 = jp.linalg.norm(tip_pos0 - cube_pos0[None, :], axis=-1)

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
            },
            info={
                'rng': rng,
                'initial_cube_pos': cube_pos0,
                'prev_tip_dists_sum': jp.sum(tip_dists0),
            },
        )

    def step(self, state, action):
        rng = state.info['rng']
        rng, _ = jax.random.split(rng)
        initial_cube_pos = state.info['initial_cube_pos']
        prev_tip_sum = state.info['prev_tip_dists_sum']

        # 텐던 motor는 이미 ctrlrange=[-1,1]인 torque control -> action 그대로 사용
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

        proximity_delta = (prev_tip_sum - tip_dists_sum) * 100.0
        reward_reach = jp.clip(proximity_delta, -2.0, 5.0) * 5.0

        motor_vel = pipeline_state.qvel[self.finger_qvel_idx]
        is_slow = jp.exp(-5.0 * jp.mean(jp.abs(motor_vel)))
        is_close = jp.clip((0.025 - min_tip_dist) / 0.025, 0.0, 1.0)
        reward_grasp = is_close * is_slow * 20.0

        cube_displacement = jp.linalg.norm(cube_pos - initial_cube_pos)
        contact_signal = jp.where(
            cube_displacement < 0.005, 0.0,
            jp.where(cube_displacement < 0.03, 1.0,
                     jp.exp(-(cube_displacement - 0.03) / 0.015))
        )
        reward_contact = contact_signal * 5.0

        # 이 손은 반경이 더 큼(~0.15m) -> 이탈 판정 반경도 그에 맞게 확대
        penalty_rolloff = jp.maximum(cube_lateral - 0.09, 0.0) * 50.0

        total_reward = (
            reward_reach + reward_grasp + reward_contact
            - penalty_rolloff - 0.1
        )

        # 큐브는 이제 높이가 고정이라 "낙하" 개념이 없음 -> 이탈(rolloff)만 종료조건으로 사용
        too_far = cube_lateral > 0.14
        done = jp.where(too_far, 1.0, 0.0)
        total_reward -= jp.where(done > 0, 100.0, 0.0)

        new_metrics = {
            'reward': total_reward,
            'reward_reach': reward_reach,
            'reward_grasp': reward_grasp,
            'reward_contact': reward_contact,
            'penalty_rolloff': penalty_rolloff,
            'min_tip_dist_per_step': min_tip_dist,
            'cube_height_per_step': cube_height,
        }

        new_info = state.info.copy()
        new_info['rng'] = rng
        new_info['prev_tip_dists_sum'] = tip_dists_sum

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=total_reward,
            done=done,
            metrics=new_metrics,
            info=new_info,
        )
