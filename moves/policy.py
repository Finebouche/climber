"""Run a learned policy (python -m climber.learning.train: outputs/learning/<run>/policy.onnx) on the robot.

A move of microban (branch dev_v2), with onnxruntime (installed on the robot).  The observation and the action are
those of climber/learning/env.py (ClimbEnv): keep both in step.  It reads ``policy.onnx`` next to it:

    cp outputs/learning/<run>/policy.onnx moves/policy.onnx
    cp moves/policy.py moves/policy.onnx ../microban/src/moves/          # then make sim WALL=N2 in microban/
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import microban_model
import numpy as np
import onnxruntime as ort
from microban_model import JOINTS

POLICY = Path(__file__).with_name("policy.onnx")
MAX_STEP = 0.05                              # rad per step for an action of 1 (ClimbEnv's max_step)


def joint_limits() -> np.ndarray:
    """19 x 2, min and max of each joint (rad), order JOINTS, from the climbing robot's URDF."""
    limits = {j.get("name"): j.find("limit") for j in ET.parse(microban_model.urdf(climb=True)).iter("joint")}
    return np.array([[float(limits[n].get("lower")), float(limits[n].get("upper"))] for n in JOINTS])


def observe(obs) -> np.ndarray:
    """ClimbEnv's observation: q, dq, gravity, gyro (NaN -> 0), shape (1, 44)."""
    x = np.concatenate([obs["q"], obs["dq"], obs["gravity"], obs["gyro"]])
    return np.nan_to_num(x).astype(np.float32)[None]


class PolicyMove:
    """Run the learned climbing policy (policy.onnx)."""

    variants = ("climber",)
    kp = 125                                 # the gain the policy was trained with (ClimbEnv: kp_fw=125)

    def __init__(self):
        self.session = ort.InferenceSession(str(POLICY)) if POLICY.exists() else None
        self.low, self.high = joint_limits().T

    def start(self, obs):
        self.target = obs["q"].copy()
        if self.session is None:
            print(f"policy: no {POLICY}")

    def step(self, obs):
        if self.session is None:
            return None
        action = self.session.run(None, {"obs": observe(obs)})[0][0]
        self.target = np.clip(self.target + MAX_STEP * action, self.low, self.high)
        return self.target
