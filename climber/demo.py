"""A first simulation to watch: the robot hanging on a wall's start holds sways its trunk left and right, leans back
from the wall, the four balls staying in their holds (the IK gives the joint targets every 20 ms), then turns its
head.  The trunk follows only part of the way: the four limbs make a closed chain and the servos (kp 400) stop short
of their targets under load; going up barely works at all (the ankles stall).

    uv run mjpython -m climber.demo [N2]          # macOS: MuJoCo's viewer needs mjpython (Linux: python)
    uv run python -m climber.demo --no-view       # without a window: prints what happened
"""
import argparse
import math

import numpy as np

from .common import JOINTS, make_sim

PERIOD_S = 3.0
SWAY = 0.03                                  # m, trunk left and right (it follows about half)
LEAN = 0.03                                  # m, trunk away from the wall

parser = argparse.ArgumentParser()
parser.add_argument("wall", nargs="?", default="N2")
parser.add_argument("--no-view", action="store_true", help="no window")
args = parser.parse_args()

sim = make_sim(args.wall)
sim.reset_hanging()
viewer = None if args.no_view else sim.view()            # every step() now shows in the viewer, at real time
balls = sim.ball_positions()
trunk_pos, trunk_quat = sim.trunk_pose()
n = int(round(PERIOD_S / sim.control_dt))
phase = [2 * math.pi * t / n for t in range(n)]
offsets = ([[SWAY * math.sin(s), 0.0, 0.0] for s in phase] * 2         # sway twice
           + [[0.0, -LEAN * (1 - math.cos(s)) / 2, 0.0] for s in phase])  # lean back and return (y < 0: away)
q = sim.joint_positions()
moved = []
for offset in offsets:
    q, _ = sim.joint_angles_for(balls, trunk_pos + offset, trunk_quat, q_start=q, iters=50)
    sim.step(q)
    moved.append(sim.trunk_pose()[0] - trunk_pos)
    if viewer is not None and not viewer.is_running():
        break
head = JOINTS.index("head")
for s in phase:                                           # the head, left and right
    target = q.copy()
    target[head] = 0.6 * math.sin(s)
    sim.step(target)
    if viewer is not None and not viewer.is_running():
        break

moved = 1000 * np.array(moved)
print(f"trunk moved x {moved[:, 0].min():+.0f} / {moved[:, 0].max():+.0f} mm (asked +-{1000 * SWAY:.0f}), "
      f"y {moved[:, 1].min():+.0f} mm (asked -{1000 * LEAN:.0f}); hooked at the end: {sim.hooked()}")
sim.close()
