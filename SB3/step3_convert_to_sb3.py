"""
3단계: brax(JAX) 정책 -> Stable-Baselines3(PyTorch) zip 변환
─────────────────────────────────────────────────────────────
구조 진단(2단계) 결과 기준으로 작성됨:
  - 정책망: 8 -> 32 -> 32 -> 32 -> 32 -> 8  (SiLU), tanh_normal 분포
  - obs: 8-dim, action: 4-dim
  - 마지막 레이어 8개 출력 중 앞 4개 = loc(평균), 뒤 4개 = raw_std (배포시 미사용)

★ 중요 ★
brax는 최종 행동에 tanh를 씌워 [-1,1]로 압축하는 'tanh_normal' 분포를 씁니다.
SB3 기본 PPO 정책은 tanh 없이 그냥 Gaussian+clip이라, 이 차이를 무시하고 변환하면
실기기에서 동작이 미묘하게 달라집니다. 그래서 action_net 출력에 tanh를 내장한
커스텀 정책 클래스(BraxTanhPolicy)를 사용합니다.

obs 정규화(mean/std)도 모델 forward 안에 고정 버퍼로 구워넣었습니다.
=> 배포 코드(core.py 등)에서 이미 obs를 정규화하고 있다면, 이 모델을 쓸 때는
   그 정규화 단계를 반드시 제거하세요 (이중 정규화 방지).

실행 전 필요:
  pip install stable-baselines3 torch gymnasium
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

INPUT_PKL = "gripper_v5_checkpoint_5898240.pkl"
OUTPUT_ZIP = "gripper_v5_sb3_converted.zip"

OBS_DIM = 8
ACT_DIM = 4
POLICY_HIDDEN = [32, 32, 32, 32]
VALUE_HIDDEN = [256, 256, 256, 256, 256]


# ─────────────────────────────────────────────────────────
# 1. brax params 로드
# ─────────────────────────────────────────────────────────
with open(INPUT_PKL, "rb") as f:
    params = pickle.load(f)

normalizer_params, policy_params, value_params = params

obs_mean = np.array(normalizer_params["mean"], dtype=np.float32)
obs_std = np.array(normalizer_params["std"], dtype=np.float32)
std_eps = float(normalizer_params["std_eps"])

print("obs_mean:", obs_mean)
print("obs_std:", obs_std)


# ─────────────────────────────────────────────────────────
# 2. obs 정규화 features_extractor + tanh를 내장한 커스텀 SB3 정책
#
#    주의: ActorCriticPolicy.get_distribution()/evaluate_actions() 내부는
#    `super().extract_features(...)`를 호출하는데, 이건 self의 서브클래스
#    오버라이드를 우회하고 곧장 BaseModel.extract_features로 갑니다.
#    그래서 extract_features를 오버라이드하는 방식으로는 정규화가 실제로는
#    적용되지 않습니다 (처음 버전의 버그 원인). 대신 SB3가 항상 제대로
#    호출해주는 features_extractor 모듈 자체를 정규화 레이어로 교체합니다.
# ─────────────────────────────────────────────────────────
class NormalizeFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, mean, std):
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        self.register_buffer("obs_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(std, dtype=torch.float32))

    def forward(self, observations):
        return (observations - self.obs_mean) / self.obs_std


class BraxTanhPolicy(ActorCriticPolicy):
    """action_net 출력에 tanh를 씌운 정책.
    deterministic=True로 predict()하면 brax의 deterministic 추론과 동일하게
    동작하도록 맞춤."""

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        mean_actions = torch.tanh(mean_actions)  # brax의 tanh 압축과 동일하게
        return self.action_dist.proba_distribution(mean_actions, self.log_std)


# ─────────────────────────────────────────────────────────
# 3. SB3 PPO 모델 생성 (구조만, 가중치는 아직 랜덤)
#    SB3 PPO는 env 없이 바로 생성이 안 되므로 최소 dummy env로 감쌉니다.
# ─────────────────────────────────────────────────────────
from stable_baselines3.common.vec_env import DummyVecEnv

obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)


class DummyGripperEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, seed=None, options=None):
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(OBS_DIM, dtype=np.float32), 0.0, True, False, {}


dummy_env = DummyVecEnv([lambda: DummyGripperEnv()])

model = PPO(
    policy=BraxTanhPolicy,
    env=dummy_env,
    policy_kwargs=dict(
        net_arch=dict(pi=POLICY_HIDDEN, vf=VALUE_HIDDEN),
        activation_fn=nn.SiLU,
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(mean=obs_mean, std=obs_std + std_eps),
    ),
)


# ─────────────────────────────────────────────────────────
# 4. 가중치 이식 (JAX -> PyTorch)
#    flax kernel shape (in, out) -> torch Linear weight shape (out, in) 이므로 transpose 필요
# ─────────────────────────────────────────────────────────
def jax_to_torch_linear(torch_linear, kernel, bias):
    w = np.array(kernel, dtype=np.float32).T  # (out, in)
    b = np.array(bias, dtype=np.float32)
    with torch.no_grad():
        torch_linear.weight.copy_(torch.tensor(w))
        torch_linear.bias.copy_(torch.tensor(b))


policy = model.policy
p = policy_params["params"]

# mlp_extractor.policy_net: hidden_0~hidden_3 (4개 은닉층)
pi_net = policy.mlp_extractor.policy_net  # nn.Sequential
pi_linears = [m for m in pi_net if isinstance(m, nn.Linear)]
for i, layer in enumerate(pi_linears):
    jax_to_torch_linear(layer, p[f"hidden_{i}"]["kernel"], p[f"hidden_{i}"]["bias"])

# action_net: hidden_4 (32 -> 8) 중 앞 4개(loc)만 사용
last_kernel = np.array(p["hidden_4"]["kernel"], dtype=np.float32)  # (32, 8)
last_bias = np.array(p["hidden_4"]["bias"], dtype=np.float32)  # (8,)
loc_kernel = last_kernel[:, :ACT_DIM]  # (32, 4)
loc_bias = last_bias[:ACT_DIM]  # (4,)
with torch.no_grad():
    policy.action_net.weight.copy_(torch.tensor(loc_kernel.T))
    policy.action_net.bias.copy_(torch.tensor(loc_bias))

# log_std는 deterministic 추론(predict(deterministic=True))에는 안 쓰이지만
# 구조상 존재해야 함 -> 0으로 둠 (필요시 raw_std 절반을 근사치로 넣어도 됨)
with torch.no_grad():
    policy.log_std.fill_(0.0)

# value_net: hidden_0~hidden_4 (5개 은닉층) + 최종 hidden_5 (256->1)
v = value_params["params"]
vf_net = policy.mlp_extractor.value_net
vf_linears = [m for m in vf_net if isinstance(m, nn.Linear)]
for i, layer in enumerate(vf_linears):
    jax_to_torch_linear(layer, v[f"hidden_{i}"]["kernel"], v[f"hidden_{i}"]["bias"])

jax_to_torch_linear(policy.value_net, v["hidden_5"]["kernel"], v["hidden_5"]["bias"])


# ─────────────────────────────────────────────────────────
# 5. 검증: 같은 obs에 대해 JAX 원본과 변환된 torch 모델의 출력이 일치하는지 확인
# ─────────────────────────────────────────────────────────
def silu(x):
    return x / (1 + np.exp(-x))


def jax_forward_reference(obs_raw):
    """brax 정책의 deterministic 추론을 순수 numpy로 재현 (검증용)."""
    x = (obs_raw - obs_mean) / (obs_std + std_eps)
    for i in range(4):
        k = np.array(p[f"hidden_{i}"]["kernel"])
        b = np.array(p[f"hidden_{i}"]["bias"])
        x = silu(x @ k + b)
    k = np.array(p["hidden_4"]["kernel"])
    b = np.array(p["hidden_4"]["bias"])
    out = x @ k + b
    loc = out[:ACT_DIM]
    return np.tanh(loc)


print("\n" + "=" * 60)
print("검증: JAX 원본 vs 변환된 PyTorch 모델 출력 비교")
print("=" * 60)
np.random.seed(0)
max_diff = 0.0
for trial in range(20):
    test_obs = np.random.uniform(-1, 1, size=OBS_DIM).astype(np.float32)

    ref_action = jax_forward_reference(test_obs)

    with torch.no_grad():
        torch_action, _ = model.predict(test_obs, deterministic=True)

    diff = np.abs(ref_action - torch_action).max()
    max_diff = max(max_diff, diff)
    if trial < 5:
        print(f"  trial {trial}: ref={ref_action}, torch={torch_action}, diff={diff:.6f}")

print(f"\n최대 오차: {max_diff:.6f}  (1e-4 이하면 정상적으로 변환된 것)")

if max_diff > 1e-3:
    print("\n⚠ 경고: 오차가 큽니다. 그대로 배포하지 말고 원인을 먼저 확인하세요.")
else:
    print("\n검증 통과. zip으로 저장합니다.")
    model.save(OUTPUT_ZIP)
    print(f"저장 완료: {OUTPUT_ZIP}")
