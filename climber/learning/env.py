"""A Gymnasium environment to climb, from the simulator's examples/rl_env.py.

* observation: what the real robot measures (joint angles and velocities, gravity in the trunk frame, gyro), as a
  policy deployed on the robot (onnxruntime, in a move) will get it;
* action: changes of the 19 joint targets per 20 ms step, in [-1, 1] times ``max_step`` rad;
* servos at kp 125, the gain the robot runs learned policies at (set ``kp = 125`` in the move);
* reward: the trunk's rise (ground truth, sim only), a bonus at the top, a penalty for a fall.  A first version: the
  reward, the start states and the curriculum (walls, seeds) are what to work on.
"""
import gymnasium as gym
import numpy as np

from ..common import JOINTS, make_sim
from ..robot import reached_top

FALL_DROP = 0.10                             # m below the start height of the trunk: a fall


class ClimbEnv(gym.Env):
    def __init__(self, wall="N2", episode_s: float = 20.0, max_step: float = 0.05):
        self.sim = make_sim(wall, learning=True)
        self.max_step = max_step
        self.max_steps = int(episode_s / self.sim.control_dt)
        n = len(JOINTS)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2 * n + 6,), np.float32)

    def _observe(self, obs):
        x = np.concatenate([obs["q"], obs["dq"], obs["gravity"], obs["gyro"]])
        return np.nan_to_num(x).astype(np.float32)          # a sensor that gave nothing reads NaN (as on the robot)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.sim.reset_hanging()
        self.target, self.steps = obs["q"].copy(), 0
        self.z0 = self.z = self.sim.trunk_pose()[0][2]
        return self._observe(obs), {}

    def step(self, action):
        low, high = self.sim.joint_range.T
        self.target = np.clip(self.target + self.max_step * np.asarray(action), low, high)
        obs = self.sim.step(self.target)
        self.steps += 1
        z = self.sim.trunk_pose()[0][2]
        reward = 10.0 * (z - self.z) - 0.01 * float(np.square(action).sum())
        self.z = z
        top, fell = reached_top(self.sim), z < self.z0 - FALL_DROP
        reward += 10.0 * top - 10.0 * fell
        info = {"height": z - self.z0, "hooked": sum(i is not None for i in self.sim.hooked().values())}
        return self._observe(obs), reward, top or fell, self.steps >= self.max_steps, info
