import gymnasium as gym
import time


env = gym.make("HalfCheetah-v5", render_mode="human")
obs, info = env.reset()

try:
    while True:
        action = env.action_space.sample()  # take a random action
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.01)
        if terminated or truncated:
            obs, info = env.reset()
except KeyboardInterrupt:
    print("사용자 종료")
finally:
    env.close()