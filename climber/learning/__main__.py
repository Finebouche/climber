"""A few random steps in ClimbEnv: the environment runs, its reward and its end of episode."""
import time

from .env import ClimbEnv

env = ClimbEnv()
obs, _ = env.reset(seed=0)
total, t = 0.0, time.time()
for k in range(200):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0.2)
    total += reward
    if terminated or truncated:
        break
print(f"{k + 1} random steps in {time.time() - t:.1f} s: return {total:.2f}, terminated {terminated}, {info}")
