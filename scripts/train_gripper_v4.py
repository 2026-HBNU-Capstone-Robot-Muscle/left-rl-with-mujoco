"""
4-finger tendon-coupled gripper grasping training - v4.0
─────────────────────────────────────────────────────────────────
v3.0(train_gripper_v3.py) 대비 변경점:
  - TendonGripperEnv(gripper_env_v3) -> TendonGripperEnv(gripper_env_v4)
  - 보상 구조: 메인(v2 스타일 reach/grasp/contact/rolloff) + 보조(팀원 설계
    R_grip/R_contact/P_slip/P_energy/P_switch, 가중치 하향)의 하이브리드
  - 큐브 도메인 랜덤화: 위치(x,y) ±1cm + 회전(yaw) ±45도(정육면체 대칭 고려)
  - 25M 스텝(v3.0)이 "접촉 0회"로 실패했던 원인(접근 유도 신호 부재)을
    reward_reach로 해결. reset/step 검증에서 실제로 접촉이 발생함을 확인함.
"""

import pickle
import os
import functools
from brax.training.agents.ppo import train as ppo
from gripper_env_v4 import TendonGripperEnv

xml_path = "/home/nam/mjx_project_gripper_v2/assets_xml/robot_box_scene.xml"
env = TendonGripperEnv(xml_path=xml_path)

CHECKPOINT_DIR = "/home/nam/mjx_project_gripper_v2/checkpoints_v4"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 중단됐던 학습을 이어가고 싶으면 아래 경로를 마지막 체크포인트 폴더로 지정
# (예: "/home/nam/mjx_project_gripper_v2/checkpoints_v4/000006553600")
# 처음부터 시작하려면 None 유지
RESTORE_FROM = None

LOG_FILE = "training_log_v4.txt"


def progress(num_steps, metrics):
    """학습 진행 모니터링. 메인 보상 + 보조 보상을 모두 출력."""
    g = lambda k: float(metrics.get(f'eval/episode_{k}',
                          metrics.get(f'eval/{k}',
                          metrics.get(k, 0.0))))

    reward = g('reward')
    reach = g('reward_reach')
    grasp = g('reward_grasp')
    contact = g('reward_contact')
    rolloff = g('penalty_rolloff')
    min_tip = g('min_tip_dist_per_step')
    grip_aux = g('R_grip_aux')
    contact_aux = g('R_contact_aux')
    slip_aux = g('P_slip_aux')
    energy_aux = g('P_energy_aux')
    switch_aux = g('P_switch_aux')
    touch = g('mean_touch_force_per_step')

    line = (
        f"S {num_steps:>9} | R {reward:8.2f} | "
        f"[main] Reach {reach:+6.2f} Grasp {grasp:5.2f} Ctct {contact:5.2f} Rolloff {rolloff:5.2f} tip {min_tip:.3f} | "
        f"[aux] Grip {grip_aux:.3f} Ctct {contact_aux:.3f} Slip {slip_aux:.3f} "
        f"Energy {energy_aux:6.2f} Switch {switch_aux:5.2f} touch {touch:.3f}"
    )
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


train_fn = functools.partial(
    ppo.train,
    num_timesteps=24_903_680,
    num_evals=39,  # 25M을 655,360의 배수(38)로 맞춰 오버슈트 최소화: 38*655360=24,903,680
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

print("=" * 90)
print("4-finger tendon gripper grasping v4.0 학습 시작 (메인+보조 보상 하이브리드)")
print("=" * 90)
print("환경 설정:")
print("  - Action space: 4-dim (right, front, left, back 텐던 motor)")
print("  - Obs space: 61-dim (motor_pos/vel 32 + cube_pos/quat/vel 13 + tip_pos 12 + touch 4)")
print("  - 큐브: 56mm 정육면체, 높이 고정, x/y 위치 + yaw 회전만 변화")
print("  - 도메인 랜덤화: cube xy ±1cm, cube yaw ±45deg (질량/크기/마찰은 고정)")
print()
print("보상 구조 (하이브리드):")
print("  [메인, v2 스타일 - 접근 유도]")
print("    + reward_reach   : 손끝-큐브 거리 감소량 기반 dense guidance")
print("    + reward_grasp   : 손끝 근접(임계값 0.06m, 56mm 큐브 기준 재조정) + 저속 시")
print("    + reward_contact : 큐브 적절히 변위(0.005~0.03m) 시")
print("    - penalty_rolloff: 큐브가 중심에서 0.09m 이상 벗어나면 페널티")
print("  [보조, 팀원 설계 - 접촉 후 다듬기, 가중치 v3 대비 1/5 수준으로 하향]")
print("    + R_grip_aux(0.2)    : 접촉 중 힘을 목표치 근처로 유지")
print("    + R_contact_aux(0.2) : 접촉 손가락 수 / 4")
print("    - P_slip_aux(0.1)    : 접촉 중 손끝 미끄러짐")
print("    - P_energy_aux(0.001): 액션 크기 합")
print("    - P_switch_aux(0.02) : 액션 부호 전환 횟수")
print()
print("학습 설정:")
print("  num_timesteps = 25M | num_envs = 1024 | batch = 1024")
print("  lr = 1e-4 | entropy = 0.01 | gamma = 0.99")
print("=" * 90)

with open(LOG_FILE, "a") as f:
    import datetime
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"학습 시작: {datetime.datetime.now().isoformat()}\n")
    f.write(f"RESTORE_FROM: {RESTORE_FROM}\n")
    f.write("=" * 80 + "\n")

make_inference_fn, params, _ = train_fn(
    environment=env,
    progress_fn=progress,
)

with open('gripper_v4_0_final.pkl', 'wb') as f:
    pickle.dump(params, f)

with open(LOG_FILE, "a") as f:
    import datetime
    f.write(f"학습 완료: {datetime.datetime.now().isoformat()}\n")

print("\n" + "=" * 90)
print("학습 완료. 모델 저장: gripper_v4_0_final.pkl")
print("=" * 90)
