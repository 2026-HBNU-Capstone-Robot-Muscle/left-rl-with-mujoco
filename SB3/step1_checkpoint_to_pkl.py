"""
1단계: brax 체크포인트(orbax 폴더) -> 단일 .pkl 파일로 변환
─────────────────────────────────────────────────────────
학습 도중 저장된 checkpoints_v5/000005898240/ 같은 폴더는 orbax 포맷이라
그대로는 pickle로 못 엽니다. 이 스크립트로 한 번 불러와서 .pkl로 다시 저장합니다.

실행 전 CHECKPOINT_DIR 경로를 실제 폴더명으로 맞춰주세요.
"""

from orbax.checkpoint import PyTreeCheckpointer
import pickle

# TODO: 실제 체크포인트 폴더 경로로 수정
CHECKPOINT_DIR = "/home/nam/mjx_project_gripper_v2/checkpoints_v5/000005898240"
OUTPUT_PKL = "gripper_v5_checkpoint_5898240.pkl"

ckptr = PyTreeCheckpointer()
params = ckptr.restore(CHECKPOINT_DIR)

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(params, f)

print(f"저장 완료: {OUTPUT_PKL}")
print("최상위 타입:", type(params))
print("길이:", len(params) if hasattr(params, "__len__") else "N/A")
