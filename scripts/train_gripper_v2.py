"""
4-finger tendon-coupled gripper grasping training - v2.1
─────────────────────────────────────────────────────────────────
v1.0(train_gripper.py) 대비 변경점:
  - GripperEnv(관절 1:1) -> TendonGripperEnv(텐던 커플링, 4관절/손가락)
  - action_size: 8 -> 4
  - xml_path: new_hand_box_scene.xml -> robot_box_scene.xml
  - 큐브가 이제 높이 고정(z=-0.08, x,y만 변화) -> "낙하" 로그 제거
"""

import pickle
import os
import functools
from brax.training.agents.ppo import train as ppo
from gripper_env_v2 import TendonGripperEnv

xml_path = "/자신의 경로 입력/mjx_project_gripper_v2/assets_xml/robot_box_scene.xml"
env = TendonGripperEnv(xml_path=xml_path)

CHECKPOINT_DIR = "/자신의 경로 입력/mjx_project_gripper_v2/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 중단됐던 학습을 이어가고 싶으면 아래 경로를 마지막 체크포인트 폴더로 지정
# (예: "/자신의 경로 입력/mjx_project_gripper_v2/checkpoints/000006553600")
# 처음부터 시작하려면 None 유지
RESTORE_FROM = None


def progress(num_steps, metrics):
    """학습 진행 모니터링. 모든 보상 항을 명시적으로 출력."""
    g = lambda k: float(metrics.get(f'eval/episode_{k}',
                          metrics.get(f'eval/{k}',
                          metrics.get(k, 0.0))))

    reward = g('reward')
    reach = g('reward_reach')
    grasp = g('reward_grasp')
    contact = g('reward_contact')
    rolloff = g('penalty_rolloff')
    min_tip = g('min_tip_dist_per_step')

    print(
        f"S {num_steps:>9} | R {reward:7.2f} | "
        f"Reach {reach:+5.2f} | Grasp {grasp:5.2f} | Ctct {contact:4.2f} | "
        f"Rolloff {rolloff:5.2f} | tip {min_tip:.3f}"
    )


train_fn = functools.partial(
    ppo.train,
    num_timesteps=25_000_000,
    num_evals=50,
    reward_scaling=1.0,
    episode_length=300,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=8,
    discounting=0.99,
    learning_rate=1e-4,
    entropy_cost=0.01,
    num_envs=1024,
    batch_size=1024,
    seed=42,
    save_checkpoint_path=CHECKPOINT_DIR,
    restore_checkpoint_path=RESTORE_FROM,
)

print("=" * 80)
print("4-finger tendon gripper grasping v2.1 학습 시작")
print("=" * 80)
print("환경 설정:")
print("  - Action space: 4-dim (right, front, left, back 텐던 motor)")
print("    손가락당 관절 4개를 텐던 coef=1.0으로 커플링, torque control")
print("  - Obs space: 57-dim")
print("    motor_pos(16) + motor_vel(16) + cube_pos(3) + cube_quat(4)")
print("    + cube_linvel(3) + cube_angvel(3) + tip_pos(4x3=12)")
print("  - 큐브: 높이(z=-0.08) 고정, x,y 위치만 변화 (자유낙하 없음)")
print("  - 도메인 랜덤화: cube_fric [1.0, 2.5] | cube_mass_scale [0.7, 1.4]")
print()
print("보상 구조:")
print("  + reward_reach: 손끝-큐브 거리 변화량 기반 접근 보상")
print("  + reward_grasp: 손끝 근접 + 저속(정지-접촉) 시 보상")
print("  + reward_contact: 큐브 적절히 변위(0.005~0.03m) 시 보상")
print("  - penalty_rolloff: 큐브가 중심에서 0.09m 이상 벗어나면 페널티")
print("  - done penalty: 큐브 이탈(0.14m 이상) 시 -100")
print()
print("학습 설정:")
print("  num_timesteps = 25M | num_envs = 1024 | batch = 1024")
print("  lr = 1e-4 | entropy = 0.01 | gamma = 0.99")
print("=" * 80)

make_inference_fn, params, _ = train_fn(
    environment=env,
    progress_fn=progress,
)

with open('gripper_v2_1_final.pkl', 'wb') as f:
    pickle.dump(params, f)

print("\n" + "=" * 80)
print("학습 완료. 모델 저장: gripper_v2_1_final.pkl")
print("=" * 80)
