"""
4-finger tendon-coupled gripper grasping training - v5.0 (실기기 스펙 대응)
─────────────────────────────────────────────────────────────────
v4.0(train_gripper_v4.py) 대비 변경점:
  - TendonGripperEnv(gripper_env_v4) -> TendonGripperEnvV5(gripper_env_v5)
  - obs: 61-dim(모터pos/vel+큐브상태+tip+touch) -> 8-dim(텐던 길이4+각속도4,
    비정규화). 실기기(DYNAMIXEL)가 실제로 볼 수 있는 정보만 정책에 노출.
  - action: 토크[-1,1] -> position 목표[-1,1](env 내부에서 텐던 목표길이로 매핑)
  - 제어주기: 물리스텝(2ms) 1개당 1 env-step -> **물리스텝 50개(=100ms)당
    1 env-step**. 실기기 명목 제어주기(100ms)와 일치시킴.
  - 보상 함수는 v4와 100% 동일 (메인 reach/grasp/contact/rolloff + 보조
    grip/contact/slip/energy/switch). obs가 줄어도 보상 계산은 특권 정보를
    계속 사용하므로 그대로 둠.

★★★ 중요: 학습 시간이 v4보다 훨씬 오래 걸립니다 ★★★
n_frames=50이라 env.step() 1번이 물리 시뮬레이션을 50배 더 계산합니다.
브랙(brax)의 "num_timesteps"는 env-step 수 기준이라 겉보기 스텝 수는 같아도
실제 물리 연산량(wall-clock 시간)은 v4의 약 50배로 늘어날 수 있습니다.
그래서 이 스크립트는 기본값을 25M이 아니라 **약 5.2M 스텝(pilot run)**으로
낮춰뒀습니다. 이 파일럿 결과(로그의 reach/grasp/contact 추세)를 먼저 보고,
방향이 맞으면 RESTORE_FROM으로 이어서 더 돌리거나 NUM_TIMESTEPS를 올려서
재실행하는 걸 권장합니다. (체크포인트에서 이어하기 가능 — 처음부터 다시
안 돌려도 됩니다.)
"""

import pickle
import os
import functools
from brax.training.agents.ppo import train as ppo
from gripper_env_v5 import TendonGripperEnvV5

xml_path = "/home/nam/mjx_project_gripper_v2/assets_xml/robot_box_scene.xml"
env = TendonGripperEnvV5(xml_path=xml_path)

CHECKPOINT_DIR = "/home/nam/mjx_project_gripper_v2/checkpoints_v5"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 중단됐던 학습을 이어가고 싶으면 아래 경로를 마지막 체크포인트 폴더로 지정
# (예: "/home/nam/mjx_project_gripper_v2/checkpoints_v5/000005242880")
# 처음부터 시작하려면 None 유지
RESTORE_FROM = None

LOG_FILE = "training_log_v5.txt"

# 655,360 = num_envs(1024) x unroll_length(20) x num_minibatches(32)
# (brax가 실제로 이 배수 단위로만 진행하므로, 오버슈트 없이 정확히 맞추려면
# num_timesteps = k * 655,360, num_evals = k+1 형태로 맞추는 게 좋음)
STEPS_PER_UNIT = 1024 * 20 * 32  # 655,360
PILOT_UNITS = 38  # 파일럿: 8 x 655,360 ≈ 5.24M 스텝. 본 학습 시 38(≈25M) 등으로 올리기.
NUM_TIMESTEPS = STEPS_PER_UNIT * PILOT_UNITS
NUM_EVALS = PILOT_UNITS + 1

# n_frames=50(=100ms/step)이라 물리 연산량이 v4 대비 50배. episode_length도
# v4의 300(=0.6s)을 그대로 쓰면 한 에피소드가 30초가 되어 너무 길어지므로,
# 실측 검증에서 grasp가 첫 100ms~수백 ms 안에 일어나는 걸 확인한 걸 기준으로
# 여유있게 50스텝(=5초)로 줄임.
EPISODE_LENGTH = 167


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
    num_timesteps=NUM_TIMESTEPS,
    num_evals=NUM_EVALS,
    reward_scaling=1.0,
    episode_length=EPISODE_LENGTH,
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
print("4-finger tendon gripper grasping v5.0 학습 시작 (실기기 스펙: obs 8-dim, position 제어)")
print("=" * 90)
print("환경 설정:")
print("  - Action space: 4-dim ([-1,1] -> 텐던 목표길이[0, 2.0944rad], position servo)")
print("  - Obs space: 8-dim (텐던 길이 4개 + 텐던 각속도 4개, 비정규화 라디안)")
print("  - 제어주기: 물리스텝 2ms x 50 = 100ms/env-step (실기기 명목 주기와 일치)")
print("  - 큐브: 56mm 정육면체, 높이 고정, x/y 위치 + yaw 회전만 변화")
print("  - 도메인 랜덤화: cube xy ±1cm, cube yaw ±45deg")
print()
print("보상 구조 (v4와 동일, obs 축소와 무관하게 특권 정보로 계산):")
print("  [메인] reward_reach / reward_grasp / reward_contact / penalty_rolloff")
print("  [보조, 가중치 하향] R_grip_aux(0.2) R_contact_aux(0.2) P_slip_aux(0.1)")
print("                      P_energy_aux(0.001) P_switch_aux(0.02)")
print()
print("학습 설정:")
print(f"  num_timesteps = {NUM_TIMESTEPS:,} (파일럿, {PILOT_UNITS}x655,360) | num_evals = {NUM_EVALS}")
print(f"  episode_length = {EPISODE_LENGTH} steps (x30ms = {EPISODE_LENGTH*0.03:.1f}s/episode)")
print("  num_envs = 1024 | batch = 1024 | lr = 1e-4 | entropy = 0.01 | gamma = 0.99")
print()
print("  ⚠ n_frames=50이라 물리 연산량이 v4 대비 약 50배입니다.")
print("    이 파일럿(5.24M 스텝) 결과를 먼저 확인한 뒤 필요시 이어서 늘리세요.")
print("=" * 90)

with open(LOG_FILE, "a") as f:
    import datetime
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"학습 시작: {datetime.datetime.now().isoformat()}\n")
    f.write(f"RESTORE_FROM: {RESTORE_FROM}\n")
    f.write(f"NUM_TIMESTEPS: {NUM_TIMESTEPS} (PILOT_UNITS={PILOT_UNITS})\n")
    f.write("=" * 80 + "\n")

make_inference_fn, params, _ = train_fn(
    environment=env,
    progress_fn=progress,
)

with open('gripper_v5_0_pilot.pkl', 'wb') as f:
    pickle.dump(params, f)

with open(LOG_FILE, "a") as f:
    import datetime
    f.write(f"학습 완료: {datetime.datetime.now().isoformat()}\n")

print("\n" + "=" * 90)
print("학습 완료. 모델 저장: gripper_v5_0_pilot.pkl")
print("결과가 괜찮으면: PILOT_UNITS를 늘리고(예: 38 -> 25M) RESTORE_FROM에")
print(f"  '{CHECKPOINT_DIR}/<마지막 스텝 폴더>' 를 넣어 이어서 학습하세요.")
print("=" * 90)
