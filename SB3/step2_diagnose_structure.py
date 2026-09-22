"""
2단계: 정책 네트워크 구조 진단
─────────────────────────────────────────────────────────
1단계에서 만든 .pkl을 열어서 레이어 구조(shape), normalizer 값,
그리고 brax가 쓰는 네트워크 생성 함수의 기본 파라미터를 출력합니다.

여기서 나온 출력 전체를 복사해서 알려주시면, 그걸 기반으로
SB3(PyTorch) zip 변환 스크립트를 정확하게 작성해드릴 수 있습니다.
"""

import pickle
import jax
import inspect
import brax.training.agents.ppo.networks as ppo_networks

INPUT_PKL = "gripper_v5_checkpoint_5898240.pkl"

with open(INPUT_PKL, "rb") as f:
    params = pickle.load(f)

print("=" * 70)
print("[1] 최상위 구조")
print("=" * 70)
print("type:", type(params))
print("길이:", len(params) if hasattr(params, "__len__") else "N/A")

if hasattr(params, "__len__"):
    for i, p in enumerate(params):
        print(f"\n--- params[{i}] 구조 (shape) ---")
        try:
            print(jax.tree_util.tree_map(lambda x: x.shape, p))
        except Exception as e:
            print(f"(tree_map 실패: {e}) 원본 출력 시도:")
            print(p)

print("\n" + "=" * 70)
print("[2] normalizer_params (params[0]) 상세 값")
print("=" * 70)
if hasattr(params, "__len__") and len(params) > 0:
    print(params[0])

print("\n" + "=" * 70)
print("[3] make_ppo_networks 시그니처 (기본값 확인용)")
print("=" * 70)
print(inspect.signature(ppo_networks.make_ppo_networks))

print("\n" + "=" * 70)
print("[4] 정책 분포(distribution) 타입 확인")
print("=" * 70)
try:
    import brax.training.distribution as distribution
    print("사용 가능한 distribution 클래스들:")
    print([x for x in dir(distribution) if "Distribution" in x])
except Exception as e:
    print("distribution 모듈 확인 실패:", e)

print("\n완료. 위 출력 전체를 복사해서 보내주세요.")
