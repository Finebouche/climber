"""Train a policy with Stable Baselines3 (PPO), several simulators in parallel processes; export it to ONNX for a
move on the robot (onnxruntime is installed there).

    python -m climber.learning.train --wall N2 --steps 2000000 --envs 8      # long: run it yourself
"""
import argparse
import json
import time

import numpy as np

from ..common import output_dir
from .env import ClimbEnv


def train(wall="N2", steps=1_000_000, envs=8, seed=0, name=None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    run = output_dir("learning", name or f"ppo_{wall}_{time.strftime('%Y%m%d_%H%M%S')}")
    venv = make_vec_env(ClimbEnv, n_envs=envs, seed=seed, env_kwargs={"wall": wall},
                        vec_env_cls=SubprocVecEnv if envs > 1 else None)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=1)
    model.learn(total_timesteps=steps)
    model.save(run / "model")
    (run / "config.json").write_text(json.dumps({"wall": wall, "steps": steps, "envs": envs, "seed": seed}))
    export_onnx(model, run / "policy.onnx")
    venv.close()
    return run


def export_onnx(model, path):
    """The deterministic policy (observation -> action in [-1, 1]) as ONNX, for onnxruntime in a move."""
    import torch

    class Policy(torch.nn.Module):            # the mean action of SB3's MlpPolicy, without its distribution
        def __init__(self, policy):
            super().__init__()
            self.features = policy.features_extractor
            self.extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs):
            latent_pi, _ = self.extractor(self.features(obs))
            return torch.clamp(self.action_net(latent_pi), -1.0, 1.0)

    obs = torch.as_tensor(np.zeros((1, *model.observation_space.shape), np.float32))
    torch.onnx.export(Policy(model.policy), obs, str(path), input_names=["obs"], output_names=["action"],
                      external_data=False)               # one file (batch of 1: one observation per step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall", default="N2")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name")
    a = parser.parse_args()
    print(train(a.wall, a.steps, a.envs, a.seed, a.name))
