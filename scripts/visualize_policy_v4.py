"""
학습된 그리퍼 정책(.pkl) 시각화 스크립트 - v4
─────────────────────────────────────────────────────────────────
사용법:
    python visualize_policy_v4.py --pkl gripper_v4_0_final.pkl --xml robot_box_scene.xml

v3(visualize_policy_v3.py) 대비 변경점:
  - from gripper_env_v3 import TendonGripperEnv
    -> from gripper_env_v4 import TendonGripperEnv
    (v3 pkl은 obs/보상 구조가 달라서 v4 pkl과 호환 안 됨 — 반드시 맞춰서 사용)
  - 롤아웃이 끝나면 메인/보조 보상 구성요소별 평균을 콘솔에 출력
    (영상만으로는 "왜 이렇게 움직였는지" 판단하기 어려워서 추가)

WSL 등 GUI 디스플레이가 없는 환경을 고려해서, 인터랙티브 뷰어 대신
mp4 영상 파일로 저장하는 방식을 씁니다 (headless EGL 렌더링).
"""

import argparse
import pickle
from collections import defaultdict

import jax
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from gripper_env_v4 import TendonGripperEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', type=str, required=True, help='학습된 파라미터 pkl 경로')
    parser.add_argument('--xml', type=str, required=True, help='씬 xml 경로')
    parser.add_argument('--out', type=str, default='rollout_v4.mp4', help='출력 영상 경로')
    parser.add_argument('--episode_length', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true', default=True)
    args = parser.parse_args()

    env = TendonGripperEnv(xml_path=args.xml)

    # 학습 때와 동일한 네트워크 구조로 재구성 (normalize_observations=True 였으므로 동일하게)
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    with open(args.pkl, 'rb') as f:
        params = pickle.load(f)

    policy = make_policy(params, deterministic=args.deterministic)
    jit_policy = jax.jit(policy)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng, act_rng = jax.random.split(rng, 3)
    state = jit_reset(reset_rng)

    # ── 롤아웃 (mjx, GPU/CPU 자동) ────────────────────────────────
    qpos_trajectory = [np.array(state.pipeline_state.qpos)]
    rewards = []
    metric_history = defaultdict(list)
    print(f'{args.episode_length} 스텝 롤아웃 시작...')
    for t in range(args.episode_length):
        act_rng, sub_rng = jax.random.split(act_rng)
        action, _ = jit_policy(state.obs, sub_rng)
        state = jit_step(state, action)
        qpos_trajectory.append(np.array(state.pipeline_state.qpos))
        rewards.append(float(state.reward))
        for k, v in state.metrics.items():
            metric_history[k].append(float(v))
        if bool(state.done):
            print(f'  {t}스텝에서 종료(done) 발생')
            break

    print(f'롤아웃 완료. 총 reward: {sum(rewards):.2f}, 평균: {np.mean(rewards):.3f}')

    # ── 보상 구성요소별 평균 (메인 / 보조) ─────────────────────────
    def avg(key):
        return np.mean(metric_history[key]) if key in metric_history else float('nan')

    print()
    print('─' * 60)
    print(f"[main] reach={avg('reward_reach'):+7.3f}  grasp={avg('reward_grasp'):7.3f}  "
          f"contact={avg('reward_contact'):7.3f}  rolloff={avg('penalty_rolloff'):6.3f}  "
          f"min_tip={avg('min_tip_dist_per_step'):.4f}")
    print(f"[aux]  grip={avg('R_grip_aux'):.4f}  contact={avg('R_contact_aux'):.4f}  "
          f"slip={avg('P_slip_aux'):.4f}  energy={avg('P_energy_aux'):7.3f}  "
          f"switch={avg('P_switch_aux'):6.3f}  touch={avg('mean_touch_force_per_step'):.4f}")
    print('─' * 60)

    # ── 렌더링 (일반 mujoco CPU 렌더러로, qpos만 재생) ────────────
    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

    cam = mujoco.MjvCamera()
    cam.distance = 0.3
    cam.azimuth = 30
    cam.elevation = -20
    cam.lookat = [0, 0, 0.12]

    frames = []
    for qpos in qpos_trajectory:
        mj_data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render().copy())

    try:
        import imageio
        imageio.mimsave(args.out, frames, fps=30)
        print(f'영상 저장 완료: {args.out}')
    except ImportError:
        # imageio 없으면 프레임을 개별 png로 저장 (fallback)
        import os
        outdir = args.out.rsplit('.', 1)[0] + '_frames'
        os.makedirs(outdir, exist_ok=True)
        for i, frame in enumerate(frames):
            mujoco.Renderer  # noqa
            from PIL import Image
            Image.fromarray(frame).save(f'{outdir}/frame_{i:04d}.png')
        print(f'imageio 없어서 프레임 png로 저장: {outdir}/ (pip install imageio 후 재실행 추천)')


if __name__ == '__main__':
    main()
