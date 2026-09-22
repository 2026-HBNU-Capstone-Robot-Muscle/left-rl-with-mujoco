"""
4단계: 완전히 새로운 프로세스에서 변환된 zip을 로드해 실제로 복원되는지 검증
─────────────────────────────────────────────────────────────
step3에서 만든 zip은 커스텀 정책 클래스(BraxTanhPolicy 등)를 쓰기 때문에,
"저장은 성공했지만 다른 프로세스/환경에서 로드가 안 되는" 상황이 있을 수
있습니다. 이 스크립트를 별도로(step3와 다른 실행으로) 돌려서 확인하세요.

이상적으로는 run.py/apply_policy.py가 실제로 쓰는 환경(Windows,
stable-baselines3 2.7.0)에서 이 스크립트를 실행해보는 게 가장 확실합니다.
"""

import hashlib
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "gripper_v5_sb3_converted.zip"

# ─────────────────────────────────────────────────────────
# 1. SHA256 출력 (config.provisional.json의 policy.sha256에 넣을 값)
# ─────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
print("SHA256:", digest)
print("-> config.provisional.json 의 cfg['policy']['sha256'] 값을 이걸로 교체하세요.\n")

# ─────────────────────────────────────────────────────────
# 2. 완전히 새 프로세스에서 로드 (커스텀 클래스 복원 여부 확인)
# ─────────────────────────────────────────────────────────
print("PPO.load() 시도 중...")
model = PPO.load(MODEL_PATH, device="cpu")
print("로드 성공.")
print("policy class:", type(model.policy))
print("observation_space:", model.observation_space)
print("action_space:", model.action_space)

assert model.observation_space.shape == (8,), "obs shape 불일치"
assert model.action_space.shape == (4,), "action shape 불일치"
assert np.all(model.action_space.low == -1) and np.all(model.action_space.high == 1), "action bound 불일치"
print("run.py의 Policy.__init__ 검증 조건 모두 통과.")

# ─────────────────────────────────────────────────────────
# 3. 임의 obs로 추론이 정상 작동하는지 (에러 없이 실행되는지) 확인
# ─────────────────────────────────────────────────────────
print("\n추론 테스트:")
for _ in range(3):
    obs = np.random.uniform(-1, 1, size=8).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"  obs={obs.round(3)} -> action={action.round(4)}")

print("\n모든 검증 통과. 이 zip은 run.py에서 정상적으로 로드/추론될 것으로 보입니다.")
